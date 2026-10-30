# Layouts and strided views

`TZoR::layout::Layout` maps an N-dimensional logical tensor index to a linear storage offset. A layout stores `shape`, signed `strides`, and a base `offset`. For a logical index `i`, the storage position is `offset + Σ(i[d] * strides[d])`. The tensor then reads that position from its backing `Storage`.

The layout is public so it can be inspected and used in low-level view construction. `Tensor::from_comp` and `Tensor::view_as` do not check that all computed positions are inside the storage allocation; construct such layouts only when their bounds are known to be valid.

## Build and inspect layouts

```rust
use TZoR::layout::Layout;

fn main() -> Result<(), TZoR::error::TensorError> {
    let row_major = Layout::contiguous(&[2, 3]);
    assert_eq!(row_major.shape(), &[2, 3]);
    assert_eq!(row_major.strides(), &[3, 1]);
    assert!(row_major.is_contiguous());

    let transposed = row_major.transpose(0, 1)?;
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.strides(), &[1, 3]);
    Ok(())
}
```

`contiguous_strides(shape)` computes row-major strides. `Layout::new(shape, strides, offset)` validates only that shape and stride ranks match. Accessors are `shape()`, `strides()`, `offset()`, `ndim()`, `numel()`, and `is_contiguous()`.

## Transform operations

| API | Behavior |
| --- | --- |
| `reshape(new_shape)` | Return a contiguous-compatible shape view with the same element count; noncontiguous layouts are rejected. |
| `permute(dims)` | Reorder axes. `dims` must be a permutation of `0..ndim`. |
| `transpose(d0, d1)` | Swap two axes. |
| `broadcast_to(target)` | Expand singleton or leading dimensions by assigning zero strides. The target rank cannot be smaller. |
| `narrow(dim, start, len)` | Restrict a dimension to a subrange, updating shape and offset. |
| `select(dim, index)` | Select a length-one range then remove the selected axis. |
| `unsqueeze(dim)` | Insert a singleton axis and choose the corresponding stride. |
| `squeeze(dim)` | Remove an axis only if its size is one. |
| `flip(dim)` | Reverse a dimension by moving the offset to its last element and negating its stride. |
| `offsets()` | Return a `StridedIter` over the storage offsets for elements in logical iteration order. |

Dimension helpers accept zero-based unsigned indices at this level. `normalize_dim(dim, ndim)` converts a valid signed dimension (including negative values) into an unsigned index. `normalize_dim_inclusive` also allows the insertion point equal to `ndim`, as needed for `unsqueeze`. Invalid dimensions return `TensorError::InvalidDim`.

`broadcast_shapes(a, b)` computes right-aligned broadcast shape. Trailing dimensions must match or one must be `1`; incompatible shapes return `ShapeMismatch`.

## Contiguity and iteration

`is_contiguous()` compares strides against row-major expectations while ignoring stride differences for size-one axes. A transposed, narrowed, broadcast, or flipped layout may not be contiguous. Broadcast axes use a stride of zero; flipped axes use negative strides.

`StridedIter` implements `Iterator<Item = usize>`. Each item is the linear offset of the next logical element. It also exposes `size_hint`, `all`, and `any` through the iterator interface. Iteration order is row-major over logical indices, regardless of the backing strides.

Tensor view methods use these layout operations to share the original storage: `reshape`, `transpose`/`t`, `permute`, `unsqueeze`, `squeeze`, `broadcast_to`/`expand`, `narrow`, `select`, and `flip`. A layout view does not copy elements. Call `Tensor::compact()` or `to_contiguous()` to materialize a contiguous copy.
