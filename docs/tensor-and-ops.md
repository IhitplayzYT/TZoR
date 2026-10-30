# Tensor and tensor operations

This guide covers `TZoR::tensor::Tensor<T>` and methods implemented across `tensor`, `ops_elementwise`, `ops_stats`, and `ops_linalg`. `Tensor<T>` has public `storage` and `layout` fields; it owns a shared backing storage handle (`Arc`) and a layout. Cloning a tensor is cheap; view methods can share the same storage. Materializing methods such as `compact` create new contiguous storage.

Methods that can fail return `Result<T, TensorError>` (or its alias `TensorResult<T>`). Dimension arguments are usually `isize`, accept negative indexing, and are checked against rank. Arithmetic between tensors uses right-aligned broadcasting when dimensions are equal or one is `1`.

## Create and inspect

```rust
use TZoR::tensor::Tensor;

fn main() -> Result<(), TZoR::error::TensorError> {
    let x = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    assert_eq!(x.shape(), &[2, 3]);
    assert_eq!(x.strides(), &[3, 1]);
    assert_eq!(x.get(&[1, 2])?, 6.0);
    assert_eq!(x.to_vec()?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}
```

Constructors and generators:

| API | Behavior |
| --- | --- |
| `from_comp(storage, layout)` | Build directly from storage and layout. Caller must ensure the layout addresses valid storage. |
| `from_1d(Vec<T>)`, `from_2d(Vec<Vec<T>>)`, `from_3d(Vec<Vec<Vec<T>>>)` | Build from nested vectors; nested dimensions must be nonempty and consistent for rank 2/3. |
| `from_vec(data, shape)` | Build a contiguous tensor; element count must equal the shape product. |
| `scalar(value)` | Rank-zero, one-element CPU tensor. |
| `fill(shape, value)`, `zeros(shape)`, `ones(shape)` | Construct a filled tensor. `zeros_like`, `ones_like`, `fill_like` match the current tensor's shape and device. |
| `arange(start, end, step)` | Half-open sequence, supporting positive or negative nonzero steps. |
| `eye(n)` | `n × n` identity matrix. |
| `from_fn(shape, closure)` | Call the closure for each row-major multi-index. |
| `linspace(start, end, steps)` | `steps` evenly spaced values including endpoints where possible. |
| `rand(shape, rng)`, `randn(shape, rng)` | Uniform `[0, 1)` or standard normal values using `Utilities::random::Random::Rng`. |

With the `cuda` feature enabled, `fill`, `zeros`, `ones`, `arange`, and `linspace` have CUDA-specific code paths for floating point types, and these paths target CUDA ordinal 0. `from_vec`, nested-vector constructors, `from_fn`, and `eye` build CPU storage. If you need CPU-only behavior, disable default features as described in the root README, or construct values with a CPU constructor and explicitly select the target device.

Inspection/data APIs: `device`, `device_str`, `dtype`, `shape`, `strides`, `offset`, `ndim`, `numel`, `is_contiguous`, `layout`, `storage`, `host_data`, `to_vec`, `item`, `get`, `len(dim)`, `downcast_f32`, `downcast_f64`, `deref_as_f32`, and `deref_as_f64`. The free helper functions `tensor::deref_as_f32(&Vec<T>)` and `tensor::deref_as_f64(&Vec<T>)` convert vectors of floating point values to the requested output type. `host_data` returns a `Cow<[T]>` and gathers noncontiguous or CUDA values to host memory. `item` requires exactly one element. `get` requires one in-bounds index per dimension.

## Device, copies and views

| API | Behavior |
| --- | --- |
| `to_device(device)`, `on(device)` | Return a tensor on the requested device. `on` is an alias. |
| `cpu()` | Copy/move to CPU. |
| `cuda(ordinal)` | Copy/move to a CUDA ordinal when compiled with CUDA. |
| `on_device_unchecked(device)` | Consuming convenience method that panics if transfer fails. |
| `scalar_like(value)`, `scalar_on(value, device)` | Create a scalar on a related or explicit device. |
| `same_device(other)` | Validate that two tensors share a device. |
| `via_host(closure)` | Run a fallible transformation through host-backed tensor data and move the result back to the original device. |
| `compact()`, `to_contiguous()` | Materialize logical elements in contiguous order. |
| `view_as(layout)` | Create a shared-storage view with the provided layout. |
| `reshape(shape: &[isize])`, `flatten()` | Change shape; `-1` can infer one dimension. Reshape requires a compatible element count and contiguous-compatible layout. |
| `transpose(d0, d1)`, `t()` | Swap dimensions; `t()` swaps the last two dimensions. |
| `permute(dims)` | Reorder dimensions. |
| `unsqueeze(dim)`, `squeeze(dim)`, `squeeze_all()` | Insert or remove singleton dimensions. |
| `broadcast_to(shape)`, `expand(shape)` | Make a zero-stride broadcast view. |
| `narrow(dim, start, len)`, `select(dim, index)` | Return a range view or a single-index view with that dimension removed. |
| `index(indices)`, `slice(indices)` | Consuming indexing helpers; `index` accepts signed indices, while `slice` uses unsigned indices. See the implementation for the sentinel rules. |
| `flip(dim)`, `movedim_last(dim)` | Reverse a dimension view or move a dimension to the last position. |
| `map(closure)`, `cast<U>()` | Elementwise conversion into a newly built tensor. `cast` converts through `f64`. |
| `allclose(other, rtol, atol)` | Compare values with relative and absolute tolerances. |

## Combine, select and reorder values

`cat(tensors, dim)` concatenates tensors along an existing dimension. `cat_with` prepends `self` to the input list. `stack(tensors, dim)` inserts a dimension and concatenates; `stack_with` also includes `self`. `split(dim, size)` splits into sections of at most `size`; `chunk(dim, chunks)` requests a number of sections. `repeat(dim, repeats)` repeats along one dimension.

Selection/indexing methods:

- `index_select(dim, indices)` gathers slices in the requested order.
- `take(indices)` gathers from a flattened tensor; `gather_flat(indices)` is the corresponding explicit name.
- `gather(dim, indices)` uses an integer index tensor (`Tensor<i64>`) to gather along a dimension.
- `masked_select(mask)` selects elements where the mask is nonzero/true according to the implementation.
- `scatter(dim, index, src)` writes source values at indices into a result.
- `argsort()` returns indices as `Tensor<i64>`; `sort()`, `sort_desc()`, `sort_by(compare)`, and `reverse()` return sorted/reversed flattened tensors.
- `unique()` returns unique values; `topk(k)` returns `(values, indices)`.

## Elementwise operations

The following operations are methods on `Tensor<T>` (from `ops_elementwise`). Most return a tensor and can fail due to incompatible broadcast shapes or device mismatch.

The public operation tags are `UnaryOp::{Neg, Abs, Exp, Ln, Sqrt, Sin, Cos, Tanh, Sigmoid, Relu, Recip, Square, Gelu, Floor, Ceil}` and `BinaryOp::{Add, Sub, Mul, Div, Max, Min, Eq, Neq, Gt, Lt, Ge, Le, Pow}`. They are passed to the generic `unary_with`, `zip_with`, and `scalar_op` kernels as descriptive tags; the supplied closure defines the computed value.

| Group | APIs |
| --- | --- |
| Generic kernels | `unary_with(op, closure)`, `zip_with(rhs, op, closure)`, `scalar_op(value, op, closure)`, `map(closure)`, `count()` |
| Arithmetic | `add`, `sub`, `mul`, `div`, `add_scalar`, `sub_scalar`, `mul_scalar`, `div_scalar`, `neg` |
| Bounds and comparisons | `abs`, `clamp(lo, hi)`, `maximum`, `minimum`, `eq_t`, `neq_t`, `gt`, `lt`, `ge`, `le`, `thresh_eq_t`, `thresh_neq_t`, `where_cond(a, b)` |
| Scalar powers | `square`, `pow(rhs)`, `powf(e)`, `powi32(n)`, `powi64(n)`, `recip` |
| Elementary functions | `exp`, `ln`, `sqrt`, `sin`, `cos`, `tanh`, `floor`, `ceil`, `sign` |
| Activations | `sigmoid`, `relu`, `lrelu(alpha)`, `rrelu(alpha)`, `prelu(alpha)`, `gelu`, `selu(lambda, alpha)`, `heavside`, `elu(alpha)`, `softplus`, `hardshrink(lambda)`, `hardtanh(min, max)`, `hardsigmoid`, `tanhshrink`, `silu`, `swish`, `softsign`, `mish` |
| Triangular masks | `tril(diagonal)`, `triu(diagonal)`, `lower_triangle`, `upper_triangle`, and alternate forms `alt_tril`, `alt_triu`, `alt_lower_triangle`, `alt_upper_triangle` |
| Predicates | `all(closure)`, `any(closure)` |

The overloaded `+`, `-`, `*`, `/` operators are available for owned and borrowed tensors where their trait implementations apply. Operator overloads call fallible methods internally and panic if the operation fails; use the named methods when errors should be propagated.

```rust
use TZoR::tensor::Tensor;

fn main() -> Result<(), TZoR::error::TensorError> {
    let x = Tensor::<f32>::from_vec(vec![-2.0, 1.0, 3.0], &[3])?;
    let y = x.relu()?.mul_scalar(2.0)?;
    assert_eq!(y.to_vec()?, vec![0.0, 2.0, 6.0]);
    Ok(())
}
```

## Reductions and statistics

Dimension reductions take `(dim: isize, keepdim: bool)` unless noted. Their `_all` variants reduce all values and usually return a scalar `T` rather than a rank-zero tensor.

| Group | APIs |
| --- | --- |
| Basic reductions | `sum`, `max`, `min`, `prod`, `mean`, `argmax`, `argmin`; `_all` variants for sum/max/min/prod/mean |
| Order statistics | `median`, `quartile(dim, Quartile)`, `percentile(p, dim, keepdim)`, `p99`, `p95`, `p90`; `_all` forms; `mode`, `mode_all` |
| Spread | `range(dim)`, `range_all`, `iqr(dim)`, `iqr_all`, `var(dim, keepdim, is_sample)`, `std(dim, keepdim, is_sample)`, `var_all(is_sample)`, `std_all(is_sample)`, `coeff_of_var`, `coeff_of_var_all` |
| Magnitude and relation | `mag`, `cov(other)`, `pearson(other)`, `corr(other)`, `cov_matrix`, `corrcoef`, `corr_mat` |
| Shape statistics | `skewness`, `kurtosis`, `skewness_all(is_sample, dim)`, `kurtosis_all(is_sample, dim)` |

`Quartile` is `Utilities::Utilities::Quartile` with `Q1`, `Q2`, `Q3`, and `Q(u8)` (the latter is clamped to 0–100). Percentile arguments use a 0–100 scale. Sample variance/std use `is_sample = true`.

`ReduceOp::{Sum, Max, Min}` selects the built-in `reduce_op`. The generic `reduce_with(dim, keepdim, closure)` gathers each reduction slice and calls the closure, returning a tensor whose element type may differ from the input. `median`, quartiles, percentiles, mode, and product use this host-backed reduction path in the current implementation.

## Linear algebra and neural network operations

| Group | APIs |
| --- | --- |
| Matrix/vector | `matmul(rhs)`, `dot(rhs)`, `cross(rhs)`, `outer(rhs)`, `diag()`, `trace()` |
| Matrix solves | `solve(rhs)`, `inv()`, `det()`; `is_linear_check`, `is_square_check`, `is_cubic_check`, `is_nthbic_check` return validated dimensions; boolean `is_linear`, `is_square`, `is_cubic`, `is_nthbic` checks |
| Convolution/pooling | `conv2d(weight, bias, stride, padding)`, `conv2d_backward`, `max_pool2d_with_indices`, `max_pool2d_backward`, `avg_pool2d`, `avg_pool2d_backward` |
| Norms/distances | `norm(p, dim, keepdim)`, `normalize(dim)`, `cosine_similarity(rhs, dim)`, `pdim_dist(other, p, dim, keepdim)`, `manhatten(other, dim)`, `euclidean(other, dim)` |
| Normalization/probability | `softmax(dim)`, `logsumexp(dim, keepdim)`, `log_softmax(dim)`, `layer_norm(eps)` |

Convolution and pooling expect image-like tensors as specified by the implementation (NCHW input and OIHW convolution weights). Stride and padding are `(height, width)` pairs. `conv2d_backward` returns gradients for input, weight and bias. Pooling backward helpers accept the output gradient and, for max pooling, the indices from the forward operation. Review `src/ops_linalg.rs` for exact shape conventions.

The module also exposes `gauss_jordan(lhs, lhs_width, rhs, rhs_width) -> Option<T>`, a low-level in-place elimination helper over flat mutable slices. It returns `None` when a pivot cannot be used, otherwise the determinant-like accumulated pivot product; callers are responsible for supplying compatible buffers and dimensions.

## Errors and behavior notes

`TensorError` variants include `ShapeMismatch`, `InvalidShape`, `InvalidDim`, `DeviceMismatch`, `Unsupported`, `Cuda`, and `Custom`. Errors are returned for invalid shapes, ranks, devices and unsupported paths. `eye`, `from_fn`, and some convenience generators unwrap internally for shapes they assume are valid. CUDA support is optional and operator coverage depends on the element type and kernel implementation.
