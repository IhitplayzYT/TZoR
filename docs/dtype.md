# Dtypes and numeric traits

The dtype module defines the numeric contracts required by tensors and lists the built-in Rust types that implement them. The library uses Rust's element type as the tensor dtype: `Tensor<f32>` stores `f32` elements, and so on. There is no runtime dtype conversion implicit in arithmetic.

```rust
use TZoR::{dtype::{Element, Float, _Element}, tensor::Tensor};

fn main() -> Result<(), TZoR::error::TensorError> {
    let x = Tensor::<f32>::from_1d(vec![1.0, 4.0, 9.0])?;
    let y = x.sqrt()?;
    assert_eq!(y.dtype(), "f32");
    assert_eq!(y.to_vec()?, vec![1.0, 2.0, 3.0]);
    Ok(())
}
```

## Built-in types

All listed types implement `_Element`. Signed and unsigned integers also implement `Element`. Only `f32` and `f64` implement `Float`.

| Rust type | `_Element` | `Element` | `Float` | CUDA type mapping |
| --- | --- | --- | --- | --- |
| `f32`, `f64` | yes | yes | yes | `float`, `double` |
| `i8`, `i16`, `i32`, `i64`, `isize` | yes | yes | no | none |
| `u8`, `u16`, `u32`, `u64`, `usize` | yes | no | no | none |

CUDA eligibility also requires the element type to satisfy `DeviceElem`. With the `cuda` feature, that blanket implementation requires `cudarc::driver::DeviceRepr + ValidAsZeroBits + Unpin`. In this implementation only floating point types advertise a CUDA type name. Device support and individual kernel coverage can vary by operation; consult [Storage](storage.md).

## Traits

### `DeviceElem`

Marker trait for element types eligible for device allocation. With CUDA disabled it has a blanket implementation for every type. With CUDA enabled its blanket implementation uses the `cudarc` device representation and zero-bit validity bounds.

### `_Element`

Base element contract used by `Tensor<T>`. It requires `Copy`, `Send`, `Sync`, `'static`, ordering, debug/display support, arithmetic and assignment arithmetic, and `DeviceElem`.

Associated constants:

- `NAME`: Rust-like type name returned by `Tensor::dtype()`.
- `CUDA_TYPE`: CUDA spelling when supported; otherwise `None`.

Methods:

- `zero`, `one`: additive and multiplicative identity.
- `lowest`, `highest`: lowest and highest representable bounds (infinity for floats).
- `from_f64`, `to_f64`, `from_f32`, `to_f32`: scalar conversion helpers. Integer conversions use Rust's `as` cast behavior.
- `max`, `min`, `clamp`: ordered scalar helpers.
- `map`: apply a scalar closure to one value.

The leading underscore is part of the public trait name.

### `Element`

Extends `_Element` with unary negation and provides:

- `abs`: absolute value by negating values below zero.
- `sign`: returns `-1` below zero and `1` otherwise, including zero.

The unsigned integer implementations do not implement this trait.

### `Float`

Extends `Element` with floating point operations. Implemented for `f32` and `f64`:

- `epsilon`
- `exp`, `ln`, `log(base)`, `sqrt`
- `sin`, `cos`, `tanh`
- `powf(exponent)`
- `square`, `cube`
- `floor`, `ceil`, `is_nan`

Tensor neural network operations, statistics that require division, and autograd are generally bounded by `Float`; integer tensors still support a broad set of construction, indexing and elementwise arithmetic operations.

## `Dtype<T>`

`Dtype<T>` is a public enum with `dim1(T)`, `dim2(T, T)` and `dim3(T, T, T)` variants. It is declared in the dtype module but is not used as the tensor's runtime dtype descriptor. Tensor rank and element type are represented separately by the `Tensor<T>` element parameter and layout shape.

## Implementing a custom element type

Implement `_Element` for custom values that satisfy its bounds, then `Element` if the type supports negation. `Float` is intended for real-valued types and requires implementations of all transcendental methods. The current constructors and operations rely on these methods for numeric conversion and arithmetic. For CUDA builds, also satisfy `DeviceElem`; custom types will need a valid device representation, and kernels may still only support built-in CUDA type names.
