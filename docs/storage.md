# Storage and devices

`TZoR::storage::Storage<T>` represents a tensor's backing allocation. A tensor combines shared storage with a `Layout`; storage holds values, while layout determines how logical indices map into that allocation. `Tensor::clone()` shares storage through `Arc`.

## Storage variants

```rust
use TZoR::tensor::Tensor;

fn main() -> Result<(), TZoR::error::TensorError> {
    let x = Tensor::<f32>::from_1d(vec![2.0, 4.0, 6.0])?;
    assert_eq!(x.storage().len(), 3);
    assert_eq!(x.storage().as_cpu(), Some(&[2.0, 4.0, 6.0][..]));
    Ok(())
}
```

With the `cuda` feature disabled, `Storage<T>` has the `Cpu(Vec<T>)` variant. When CUDA is enabled it also has `Cuda(CudaStorage<T>)`.

| API | Behavior |
| --- | --- |
| `len()` | Number of allocated elements in the storage, not necessarily the number of values in a view. |
| `is_empty()` | Whether allocation length is zero. |
| `device()` | Returns `Utilities::device::Device::Device::Cpu` or `Cuda(ordinal)`. |
| `as_cpu()` | Returns `Some(&[T])` for CPU storage and `None` for CUDA storage. |
| `as_ptr()` | CPU allocation pointer; returns a null pointer for CUDA storage. It is an allocation pointer, not a logical view pointer. |
| `Tensor::storage()` | Borrow the storage handle. |
| `Tensor::host_data()` | Return logical tensor values on host, borrowing contiguous CPU data when possible and otherwise creating an owned copy. |
| `Tensor::to_vec()` | Copy logical values into a `Vec<T>` on host. |

`Storage::Cpu` and `Storage::Cuda` are public enum variants, but directly constructing custom `Storage` and `Layout` pairs is low level. `Tensor::from_comp` does not validate that a layout's reachable offsets fit the allocation.

## Device API

The companion crate's device enum is `Utilities::device::Device::Device`:

```rust
use TZoR::tensor::Tensor;
use Utilities::device::Device::Device;

fn main() -> Result<(), TZoR::error::TensorError> {
    let x = Tensor::<f32>::from_vec(vec![1.0; 4], &[2, 2])?;
    let cpu_x = x.to_device(Device::Cpu)?;
    assert!(cpu_x.device().is_cpu());
    Ok(())
}
```

Because this example names the companion crate directly, add `Utilities = { path = "/path/to/TZoR/Utilities" }` alongside `TZoR` in the application's dependencies, as described in [Utilities](utilities.md).

Variants are `Cpu` and `Cuda(usize)`. Methods `is_cpu()` and `is_cuda()` inspect the kind. Display formatting is `cpu` and `cuda:<ordinal>`.

Tensor transfer helpers:

- `to_device(device)` and alias `on(device)` return a fallible transfer.
- `cpu()` returns a CPU copy.
- `cuda(ordinal)` requests a CUDA ordinal; it only works when the crate is built with the `cuda` feature and a compatible runtime/device setup is available.
- `on_device_unchecked(device)` consumes the tensor and panics if `on` fails. Prefer the fallible methods in application code.
- `same_device(other)` checks for a device mismatch before joint operations.
- `scalar_on(value, device)` and `scalar_like(value)` create scalar tensors on a selected or matching device.

Some operations call `via_host` and perform work on CPU before returning a tensor to its original device. Host transfers can be expensive. Consult the operation descriptions and benchmark the actual workloads before relying on device residency for performance.

## Features and element support

`cuda` is a default feature in the current manifest. To build a CPU-only dependency, disable default features:

```toml
TZoR = { path = "/path/to/TZoR", default-features = false }
```

The `dtype::Float` implementations for `f32` and `f64` provide the CUDA type identifiers `float` and `double`. Integer element types do not currently expose CUDA type identifiers. CUDA support also depends on external driver/runtime availability and on the relevant kernel being implemented.

## Pointer and lifetime caution

`as_ptr()` exposes a raw pointer to CPU allocation memory and gives no ownership or mutation guarantees beyond the borrowed storage's lifetime. For a sliced, transposed, flipped or broadcast view, pointer arithmetic alone does not describe the logical tensor; use the view's layout metadata and `host_data`/`to_vec` when a logical sequence is needed. A CUDA storage pointer is not exposed through this method.
