---
layout: post
title:  "What happened when a user donate_arg in jax.jit?"
date:   2025-09-01 0:00:00 +0900
categories: jax MLsys
---

In `pjit.py`, donated arguments are save into `donated_invars`

```python
  if (ji.donate_argnums or ji.donate_argnames) and not config.debug_nans.value:
    donated_invars = donation_vector(ji.donate_argnums, ji.donate_argnames, in_tree)
  else:
    donated_invars = (False,) * len(explicit_args)

```

In `pjit.py`, `donated_invars` is passed to `_pjit_impl`

In XLA

https://openxla.org/xla/aliasing#defining_aliasing_at_runtime

The aliasing defined in the previous step is specified during compilation. During execution, you can use the LocalClient::RunAsync API to choose whether to donate the buffer.

Input buffers to the program are wrapped in ExecutionInputs, which in turn contain a tree of MaybeOwningDeviceMemory. If memory is specified as owning (ownership of the buffer is passed to the XLA runtime), the buffer is actually donated, and the update is executed in place, as requested by the compile-time aliasing API.

If, however, the buffer that is aliased at compile time is not donated at runtime, copy-protection kicks in: an extra output buffer O is allocated, and the contents of the input buffer P that was meant to be aliased are copied into O (so effectively the program can execute as if the buffer O was donated at runtime).


https://github.com/openxla/xla/blob/main/xla/service/executable.h

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