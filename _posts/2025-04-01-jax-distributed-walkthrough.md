---
layout: post
title:  "jax.distributed and XLA Coordination Walkthrough"
date:   2025-04-01 0:00:00 +0900
categories: jax MLsys
---

*Pinned commits: JAX 0.4.35 (81991d8); XLA (76da730).*

## Use Case

```python
jax.distributed.initialize(
    coordinator_address="192.168.0.1:1234",
    num_processes=2,
    process_id=RANK
)
jitted_function()
jax.distributed.shutdown()
```

## What happens under the hood of `jax.distributed.initialize()`?

## Coordination Service

`jax.distributed.initialize()`

- **Process 0 starts the service**
  by `xla_extension.get_distributed_runtime_service` in `jax/_src/distributed.py`

```python
self.service = xla_extension.get_distributed_runtime_service(
    coordinator_bind_address, num_processes
)
```

- → `get_distributed_runtime_service` in `xla/python/xla.cc`
- → `GetDistributedRuntimeService` in `xla/pjrt/distributed/distributed.cc`
- → `DistributedRuntimeService` in `xla/pjrt/distributed/service.cc`
- → `CoordinationServiceImpl` in `xla/pjrt/distributed/service.cc`

Creates the following two services:
- `CoordinationServiceStandaloneImpl` (registered for `tsl::CoordinationServiceInterface`) at
  `xla/tsl/distributed_runtime/coordination/coordination_service.cc`
- `tsl::GrpcCoordinationServiceImpl` in
  `/xla/tsl/distributed_runtime/rpc/coordination/grpc_coordination_service_impl.h`

## Coordination Client (Agent)

`jax.distributed.initialize()`

- **All processes start clients**
  by `xla_extension.get_distributed_runtime_client` in `jax/_src/distributed.py`

```python
self.client = xla_extension.get_distributed_runtime_client(
    coordinator_address, process_id, init_timeout=initialization_timeout
)
```

- → `get_distributed_runtime_client` in `xla/python/xla.cc`
- → `GetDistributedRuntimeClient` in `xla/pjrt/distributed/client.cc`
- → `DistributedRuntimeCoordinationServiceClient` in `xla/pjrt/distributed/client.cc`
- → `coord_agent_->Initialize(options.env, "jax_worker", options.node_id, config, std::move(leader_client), error_fn);`
  in `xla/pjrt/distributed/client.cc`
  - `task_id = node_id`

- → `CoordinationServiceAgentImpl` in
  `xla/tsl/distributed_runtime/coordination/coordination_service_agent.cc`
- → `CoordinationServiceAgentImpl::Initialize`

### Client Connection Flow

```python
self.client.connect()
```

- → `CoordinationServiceAgentImpl::Connect()` in `coordination_service_agent.cc`
- → `leader_client_->RegisterTaskAsync(req, ...)` in `coordination_service_agent.cc`
  (in commit 76da730, the Async variant is replaced by a callback `done(RegisterTask)` in the RPC handler)
- → `CoordinationServiceStandaloneImpl::RegisterTask` in `coordination_service.cc`
- → `task_cluster_state->SetConnected(incarnation)`
  - `task_cluster_state` is the value for key `"jax_worker/NODE_ID"`
  - `incarnation` is a random number that uniquely identifies the agent instance and differs across restarts (e.g., a standby agent reconnecting)

```cpp
void CoordinationServiceStandaloneImpl::TaskState::SetConnected(
    uint64_t task_incarnation
) {
    state_ = CoordinatedTaskState::TASKSTATE_CONNECTED;
    status_ = absl::OkStatus();
    task_incarnation_ = task_incarnation;
    absl::MutexLock l(&last_heartbeat_mu_);
    last_heartbeat_us_ = Env::Default()->NowMicros();
}
```

## Shutdown Flow

`jax.distributed.shutdown()`

- → `self.client.shutdown()` in `jax/_src/distributed.py`
- → `CoordinationServiceAgentImpl::ShutdownInternal()` in
  `xla/tsl/distributed_runtime/coordination/coordination_service_agent.cc`
  - Sends rpc request → leader_client → `CoordinationServiceStandaloneImpl::ShutdownTaskAsync`
  - Client will shutdown regardless of service success

- → `self.service.shutdown()` in `jax/_src/distributed.py`