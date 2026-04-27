---
layout: post
title:  "What happens when you donate an argument in jax.jit?"
date:   2025-09-01 0:00:00 +0900
categories: jax MLsys
---

## JAX Side

In [`pjit.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pjit.py), donated arguments are saved into `donated_invars`:

```python
if (ji.donate_argnums or ji.donate_argnames) and not config.debug_nans.value:
    donated_invars = donation_vector(ji.donate_argnums, ji.donate_argnames, in_tree)
else:
    donated_invars = (False,) * len(explicit_args)
```

From there it gets passed down through:

- → `_pjit_call_impl_python` in `jax/_src/pjit.py`
- → `_resolve_and_lower` → `_pjit_lower` in `jax/_src/pjit.py`
- → `lower_sharding_computation` → `_cached_lowering_to_hlo` in `jax/_src/interpreters/pxla.py`
- → `lower_jaxpr_to_module` in `jax/_src/interpreters/mlir.py`

The interesting work is inside `_set_up_aliases` (`mlir.py:1431`). For every donated arg, it looks for an output with matching shape, dtype, memory kind, sharding, and layout. The function returns two lists:

- `input_output_aliases`: donated args paired with an output. Lowered as `mhlo.buffer_alias` on the MLIR function (`mlir.py:1832`).
- `xla_donated_args`: donated args without a match. Lowered as `jax.buffer_donor` on the function argument (`mlir.py:1822`), which hands the aliasing decision to the XLA compiler.

If neither path works, you get the warning at `mlir.py:1343`:

```text
Some donated buffers were not usable: <aval>.
```

## XLA Side

From the [XLA aliasing docs](https://openxla.org/xla/aliasing#defining_aliasing_at_runtime):

> The aliasing defined in the previous step is specified during compilation. During execution, you can use the `LocalClient::RunAsync` API to choose whether to donate the buffer.
>
> Input buffers to the program are wrapped in `ExecutionInput`s, which in turn contain a tree of `MaybeOwningDeviceMemory`. If memory is specified as owning (ownership of the buffer is passed to the XLA runtime), the buffer is actually donated, and the update is executed in place, as requested by the compile-time aliasing API.
>
> If, however, the buffer that is aliased at compile time is not donated at runtime, copy-protection kicks in: an extra output buffer O is allocated, and the contents of the input buffer P that was meant to be aliased are copied into O (so effectively the program can execute as if the buffer O was donated at runtime).

The three ownership states are documented in [`xla/service/executable.h`](https://github.com/openxla/xla/blob/main/xla/service/executable.h):

```cpp
// ExecutionInput buffers are in one of three states:
//
// 1) Owned by the caller and immutable.
// 2) Donated by the caller but returned on error.
// 3) Donated by the caller and freed on error.
//
// Case (1) buffers are stored as MaybeOwningDeviceMemory(DeviceMemoryBase).
// Case (2) buffers are stored as MaybeOwningDeviceMemory(OwningDeviceMemory),
//   with their indices present in unowned_indices_.
// Case (3) buffers are stored as MaybeOwningDeviceMemory(OwningDeviceMemory),
//   with their indices absent from unowned_indices_.
```

### Which case do JAX donations actually hit?

The runtime path is in [`pjrt_stream_executor_client.cc`](https://github.com/openxla/xla/blob/main/xla/pjrt/pjrt_stream_executor_client.cc). When the HLO has an entry in `input_output_alias_config` and the result buffer shares memory with the input, the input gets `is_donated = true` (around line 1605). The buffer-build loop right below it wraps every `is_donated` input as an owning `ScopedDeviceAddress` and calls `SetUnownedIndex` (line 1620). That's case 2.

So in practice JAX only produces case 1 (input not donated) or case 2 (donated and aliased). Case 3 — donated but with the index *missing* from `unowned_indices_` — is reachable from non-PJRT XLA callers like `LocalClient::RunAsync`, but it isn't what JAX emits. From Python, the case-2-vs-3 split is invisible; what you can see is whether donation was honored at all, or whether you ended up in case 1 with copy-protection silently making a copy.

## How to verify donation fired

Donation only saves memory if the aliasing actually holds. If the output doesn't match the donated input on shape, dtype, memory kind, sharding, or layout, one of three things will happen:

1. JAX gives up early and prints `Some donated buffers were not usable`.
2. JAX hands the donation to XLA via `jax.buffer_donor`, but XLA also can't find a target. The runtime falls back to copy-protection, allocates a fresh output, and copies the input data in.
3. The donation succeeds and the output reuses the input's storage.

If you want to know which one you got, dump the compiled HLO:

```python
jax.jit(f).lower(...).compile().as_text()
```

Look for `input_output_alias=` on the donated arg. If it's there, donation fired. If not, you're paying for a copy you may not have realized you were paying for.
