# Automatic differentiation

`TZoR::autograd::Comp<T>` wraps a floating point `Tensor<T>` in a reverse-mode computational graph. Operations on `Comp` record parent nodes and backward closures. Call `backward()` on a scalar result to accumulate gradients in leaves, then read them with `grad()`.

```rust
use TZoR::{autograd::Comp, tensor::Tensor};

fn main() -> Result<(), TZoR::error::TensorError> {
    let x = Comp::leaf(Tensor::<f32>::from_1d(vec![2.0, 3.0])?);
    let loss = x.square()?.sum_all()?;
    loss.backward()?;

    assert_eq!(loss.item()?, 13.0);
    assert_eq!(x.grad().unwrap().to_vec()?, vec![4.0, 6.0]);
    Ok(())
}
```

For non-scalar outputs, use `backward_with(gradient)` to provide the upstream gradient. Gradients accumulate; call `zero_grad()` before a new accumulation cycle when needed. Gradients are stored as `Option<Tensor<T>>`, so `None` means no gradient has been accumulated.

## Construct and inspect graph values

| API | Behavior |
| --- | --- |
| `Comp::new(data, requires_grad)` | Create a graph node with an explicit gradient flag. |
| `Comp::leaf(data)` | Create a leaf requiring gradients. |
| `Comp::constant(data)`, `Comp::non_leaf(data)` | Create graph values that do not require gradients. Despite the `non_leaf` name, this constructor has no parents and is considered a leaf by `is_leaf()`. |
| `Comp::from_op(output, parents, backward)` | Build a custom graph node; backward closure returns one optional gradient per parent. |
| `value.un(output, backward)` | Build a unary operation node. |
| `grad()` | Clone the accumulated gradient, if present. |
| `zero_grad()` | Clear the accumulated gradient. |
| `requires_grad()`, `is_leaf()` | Inspect graph state. |
| `data()` | Clone the wrapped tensor. |
| `set_data(tensor)` | Replace wrapped data after shape/device compatibility checks. |
| `shape()`, `ndim()`, `numel()`, `device()` | Inspect tensor metadata. |
| `item()`, `to_vec()` | Read tensor values. |
| `detach()` | Return a constant wrapper around the same data values, disconnected from the graph. |
| `to_device(device)` | Transfer data while returning a new graph value. |
| `lin(closure)` | Apply a tensor transformation and reuse the same closure to transform gradients. Use only for linear maps whose transpose action is represented by that closure; general nonlinear maps need a correct custom derivative. |

## Graph execution and gradient mode

- `backward()` seeds a scalar output with a unit gradient and traverses the graph in reverse topological order. The output must contain one value.
- `backward_with(gradient)` supplies an explicit output gradient, which must match the output shape/device expectations.
- `topo_sort()` returns graph nodes in topological order.
- `acc_grad(gradient)` accumulates a gradient on a node.
- `is_grad_enabled()` reports the current thread's gradient recording state.
- `no_grad(closure)` temporarily disables graph recording for the closure and restores the prior state when it exits, including during unwinding.
- `with_grad(closure)` temporarily enables graph recording and restores the prior state when it exits.
- `sgd(lr)` updates the value in place using its currently accumulated gradient and the provided learning rate. It does not clear the gradient automatically.

## Differentiable operation catalog

The following are methods on `Comp<T>` and return `Result<Comp<T>>` unless shown otherwise.

### Arithmetic and comparisons

`add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `pow`, `powf`, `powi32`, `powi64`, and `where_cond(condition, a, b)` are provided. Binary elementwise operations broadcast and their gradients are reduced back to each original input shape.

`eq_t`, `neq_t`, `gt`, `lt`, `ge`, `le`, `thresh_eq_t`, and `thresh_neq_t` return constant comparison results; they do not differentiate through the comparison. Scalar arithmetic methods are `add_scalar`, `sub_scalar`, `mul_scalar`, and `div_scalar`. Unary arithmetic includes `neg`, `abs`, `square`, `clamp`, `exp`, `ln`, `sqrt`, `sin`, `cos`, `tanh`, `recip`, `floor`, and `ceil`.

Operator overloads `+`, `-`, `*`, `/` are available for `Comp` and references. Like Tensor operator overloads, they panic if an underlying fallible operation returns an error; explicit methods allow `?` propagation.

### Activations and elementwise transforms

`sigmoid`, `relu`, `lrelu(alpha)`, `rrelu(alpha)`, `prelu(alpha)`, `gelu`, `selu(lambda, alpha)`, `heavside`, `elu(alpha)`, `softplus`, `hardshrink(lambda)`, `hardtanh(min, max)`, `hardsigmoid`, `tanhshrink`, `silu`, `swish`, `softsign`, and `mish` are available. Triangular transforms are `tril(diagonal)`, `triu(diagonal)`, `lower_triangle`, `upper_triangle`, plus `alt_tril`, `alt_triu`, `alt_lower_triangle`, and `alt_upper_triangle`.

### Shape and indexing operations

`reshape`, `flatten`, `transpose`, `t`, `permute`, `unsqueeze`, `squeeze`, `squeeze_all`, `broadcast_to`, `expand`, `flip`, `narrow`, `select`, `split`, `cat`, `cat_with`, `stack`, `stack_with`, and `index_select` propagate gradients through views/gathers or corresponding reverse scatter/split operations.

### Reductions and statistics

`sum`, `mean`, `max`, `min`, `prod`, `sum_all`, `mean_all`, `max_all`, `min_all`, `prod_all`, `range`, `range_all`, `percentile`, `p99`, `p95`, `p90`, `percentile_all`, `p99_all`, `p95_all`, `p90_all`, `median`, `median_all`, `quartile`, `quartile_all`, `iqr`, `iqr_all`, `var`, `std`, `var_all`, `std_all`, `coeff_of_var`, `coeff_of_var_all`, `mag`, `cov`, `pearson`, `corr`, `cov_matrix`, `corrcoef`, `corr_mat`, `skewness`, `kurtosis`, `skewness_all`, and `kurtosis_all` are implemented. Reduction methods generally take an `isize` dimension and `keepdim` flag. Extrema ties share the gradient equally.

### Linear algebra and normalization

`matmul_nd`, `matmul`, `dot`, `cross`, `outer`, `diag`, `trace`, `inv`, `det`, `solve`, `norm`, `normalize`, `cosine_similarity`, `pdim_dist`, `manhatten`, `euclidean`, `softmax`, `logsumexp`, `log_softmax`, and `layer_norm` are available. Matrix and norm gradients are implemented in the corresponding backward closures.

## Backward helpers

The module also exports helper functions used to align gradient shapes and construct reverse-operation data:

- `unbroadcast(gradient, input_shape)`: sum away added leading axes and expanded singleton axes.
- `expand_to(gradient, shape)`: broadcast and materialize to the requested shape.
- `keep(gradient, dim, keepdim)`: restore a reduced dimension when needed.
- `rows_to_tensor(data, shape, dim)`: reconstruct tensor data after moving a dimension to the last axis.
- `split_ratio(a, b, greater)`: create an even split mask for max/min gradient routing.

These are public but primarily serve operation implementations and custom backward functions.

## Gradient and API limitations

Autograd currently requires `T: Float`, so integer tensors cannot be wrapped in `Comp`. Comparisons are nondifferentiable constants. Nonsmooth functions use the derivatives encoded by their implementation (for example, floor and ceil return zero gradients). `no_grad`/`with_grad` state is thread-local. Some convenience percentiles should be checked against their explicit `percentile` equivalent: the current `Comp::p95()` and `Comp::p90()` implementations call percentile 99, so use `percentile(95.0, ...)` or `percentile(90.0, ...)` for those values.
