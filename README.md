# TZoR 

`TZoR` is a Rust tensor library with typed numeric elements, strided views, CPU operations, optional parallel execution and optional CUDA support. The repository also contains a small companion crate named `Utilities`.

This README is the entry point. Detailed API references and examples live in the module guides:

| Guide | Contents |
| --- | --- |
| [Dtypes](docs/dtype.md) | Supported element types and numeric traits |
| [Tensor and tensor operations](docs/tensor-and-ops.md) | Construction, data access, views, indexing, arithmetic, reductions, linear algebra and neural network operations |
| [Storage](docs/storage.md) | CPU/CUDA storage, device transfers and pointers |
| [Layout](docs/layout.md) | Shape, strides, offsets, views, broadcasting and iteration |
| [Autograd](docs/autograd.md) | Computational graphs, gradients, differentiable operations and optimization |
| [Utilities](docs/utilities.md) | Device descriptors, random generator, logging, marker traits and utility contracts |
| [Support modules](docs/support-modules.md) | Errors, display formatting, parallel helpers and the low-level CUDA backend |

## Install
Add it to a Rust project by path:

```toml
[dependencies]
TZoR = { path = "/path/to/TZoR" }
```

For a project inside this repository, a relative path works as well:

```toml
[dependencies]
TZoR = { path = "../TZoR" }
```

The crate is named `TZoR` in Cargo and imported as `TZoR` in Rust. It re-exports modules rather than defining a prelude, so import the APIs you use:

```rust
use TZoR::{autograd::Comp, tensor::Tensor};
```

### Features

The default feature set enables `parallel` and `cuda`.

| Feature | Effect |
| --- | --- |
| `parallel` | Enables Rayon-backed parallel operations |
| `cuda` | Enables CUDA storage and kernels through `cudarc` |

To build without default features, for example for a CPU-only application:

```toml
[dependencies]
TZoR = { path = "/path/to/TZoR", default-features = false }
```

Enable only parallel CPU operations with:

```toml
TZoR = { path = "/path/to/TZoR", default-features = false, features = ["parallel"] }
```

CUDA is an optional feature, but it is enabled by default in this package. CUDA runtime, driver, and compiler availability depend on the system and the selected `cudarc` configuration. Disable default features when CUDA is unavailable. See [Storage](docs/storage.md) for device behavior and transfer examples.

## First example

```rust
use TZoR::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::<f32>::from_2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
    let b = Tensor::<f32>::from_2d(vec![vec![1.0, 1.0], vec![1.0, 1.0]])?;
    let sum = a.add(&b)?;

    println!("shape: {:?}", sum.shape());
    println!("values: {:?}", sum.to_vec()?);
    println!("total: {}", sum.sum_all()?);
    Ok(())
}
```

Operations that may fail return `TZoR::error::TensorError` through the crate's `Result`/`TensorResult` aliases. Use `?` to propagate shape, dimension, device, unsupported-operation, and CUDA errors. Some convenience constructors and explicitly unchecked helpers may panic; their behavior is documented in the relevant guide.

## Common workflows

- Create tensors with `Tensor::from_vec`, `from_1d`/`from_2d`/`from_3d`, `scalar`, `zeros`, `ones`, `arange`, or `from_fn`.
- Inspect metadata with `shape`, `strides`, `ndim`, `numel`, `dtype`, `device`, and `is_contiguous`.
- Convert data to host values with `to_vec` or extract one value with `item`.
- Use tensor methods for elementwise operations, shape transformations, indexing, reductions and linear algebra. Full method catalogs are in [Tensor and tensor operations](docs/tensor-and-ops.md).
- For gradients, wrap floating point tensors in `Comp`, build a graph, call `backward`, and read `grad`; see [Autograd](docs/autograd.md).

## Repository map

```text
src/                 Main tensor crate
  dtype.rs           Element and floating-point traits
  tensor.rs          Tensor representation and constructors/views
  storage.rs         Backing storage
  layout.rs          Shape, stride, offset and iteration logic
  ops_elementwise.rs Elementwise operations and activations
  ops_stats.rs       Reductions and statistical operations
  ops_linalg.rs      Linear algebra and neural network kernels
  autograd.rs        Reverse-mode automatic differentiation
Utilities/           Companion utility crate
docs/                 Detailed module guides
```

The guides describe the current public API implemented in this repository. For code-level details, the source modules above are the authoritative reference.
