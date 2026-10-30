# Support modules

This guide records the remaining public support APIs in the root crate: errors, display formatting, parallel helpers and CUDA backend functions. Most CUDA functions are implementation building blocks used by `Tensor` operations; normal application code should prefer the higher-level methods in [Tensor and tensor operations](tensor-and-ops.md).

## Errors

`TZoR::error` exports `TensorError`, `Result<T>` and `TensorResult<T>`. Both aliases are `std::result::Result<T, TensorError>`.

| Variant | Meaning |
| --- | --- |
| `ShapeMismatch { op, lhs, rhs }` | Operation could not reconcile operand shapes. |
| `InvalidShape(message)` | Invalid shape, layout or indexing request. |
| `InvalidDim { dim, ndim }` | Dimension is outside the valid range for the rank. |
| `DeviceMismatch { lhs, rhs }` | Operation requires tensors on a shared device. |
| `Unsupported(message)` | Requested type or operation is unsupported. |
| `Cuda(message)` | CUDA setup, compilation or execution failed. |
| `Custom(message)` | Operation-specific error. |

`TensorError` implements `Debug`, `Display`, and `std::error::Error`.

## Display formatting

`Tensor<T>` implements `Display` and `Debug`. Formatting prints a nested representation followed by shape, dtype, and device metadata. Large axes are abbreviated in the formatted representation. `Tensor::rprint(&tensor)` writes an untruncated nested representation to standard output and returns `Result<(), Box<dyn Error>>`; it may transfer CUDA values to host first.

```rust
use TZoR::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Tensor::<i32>::from_2d(vec![vec![1, 2], vec![3, 4]])?;
    println!("{x}");
    Tensor::rprint(&x)?;
    Ok(())
}
```

## Parallel helpers

`TZoR::par` provides slice/range helpers. When the `parallel` feature is enabled, these use Rayon; without it, they run serially.

| API | Behavior |
| --- | --- |
| `map_slice(src, f)` | Map over a borrowed slice into a `Vec`. |
| `map_range(len, f)` | Map indices `0..len` into a `Vec`. |
| `zip_slices(a, b, f)` | Zip corresponding slice values through a closure. |
| `for_each_row(out, row_len, work_per_row, f)` | Mutably process output in row chunks, partitioning rows for parallel work. |
| `reduce(data, init, f)` | Fold values from an initial accumulator. |
| `sort(data)`, `sort_by(data, compare)`, `reverse(data)` | Mutate a slice in order. |
| `argmin(data)`, `argmax(data)` | Return the index of an extremum. |

These helpers are public but expose execution details used by tensor kernels. Generic closures and data types must satisfy the listed thread-safety bounds in their Rust signatures.

## CUDA backend

The `TZoR::cuda` module exists only with the `cuda` Cargo feature. It depends on `cudarc`, a compatible CUDA driver/device, and NVRTC for runtime kernel compilation. The module exposes:

- `cu(error)`: convert a displayable error to `TensorError::Cuda`.
- `Gpu`: shared context and stream wrapper. `ctx` and `stream` are public; loaded modules are cached internally.
- `CudaStorage<T>`: device buffer with public `data`, `dev`, and `ordinal`; methods are `len()`, `from_host(slice, ordinal)`, and `to_host()`.
- `func(device, kernel_name)`: compile/load the dtype's kernel module and resolve a function.
- `cstore(tensor)`: borrow CUDA storage or return a device mismatch error.
- `wrap(data, like, shape)`: wrap a CUDA buffer as a contiguous tensor on the same device. Caller must supply a matching element count.
- `compact(tensor)`: materialize a strided device view into a dense buffer.
- `index_select(tensor, dim, indices, output_shape)` and `gather(tensor, dim, indices, output_shape)`: device gather helpers.
- `sort`, `sort_by`, `sort_desc`: device merge sort for floating point tensors.
- `unary(tensor, UnaryOp)`, `binary(a, b, BinaryOp, output_shape)`, `scalar_op(tensor, scalar, op_code)`: low-level elementwise kernels.
- `matmul(a, b, batch_shape, m, k, n)`: low-level matrix multiplication kernel.
- `reduce(tensor, dim, ReduceOp, keepdim)`: sum/min/max kernel reduction.
- `clamp(tensor, lo, hi)`, `where_cond(condition, a, b, output_shape)`: low-level selection kernels.
- `fill(shape, value, ordinal)`, `arange(start, end, step, ordinal)`: create device tensors.
- `softmax(tensor, dim)`, `layer_norm(tensor, epsilon)`: device normalization kernels.
- `masked_select(tensor, mask)`, `scatter(tensor, dim, indices, source)`: device selection and indexed update helpers.

The `cuda::kernels::source(ctype)` helper generates CUDA source for an element type; the `kernels` module itself is private, so that function is not reachable as part of the external Rust API. Direct calls to the public backend functions require CUDA-specific types from `cudarc`; use `Tensor` methods for ordinary tensor work.

## Module exports

The root crate declares `display`, `dtype`, `error`, `tensor`, `storage`, `layout`, `par`, `ops_elementwise`, `ops_linalg`, `ops_stats`, and `autograd`, plus feature-gated `cuda`. It also declares a root `random` module, but there is no `src/random.rs` or `src/random/mod.rs` in the current checkout, so that declaration cannot currently resolve; use `Utilities::random::Random::Rng` from the companion crate instead. APIs in the operation modules are implemented as inherent `Tensor` methods and are cataloged in [Tensor and tensor operations](tensor-and-ops.md).
