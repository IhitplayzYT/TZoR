# Utilities

The repository includes a companion package, `Utilities`, and the main `TZoR` package also exposes a nested `TZoR::Utilities::Utilities` contract module. The companion package provides device descriptors, a small random number generator, marker traits and a logger. The nested contract module defines traits describing shape, storage, layout, tensor, and statistical/vector operations.

## Add the companion crate

`TZoR` already depends on `Utilities` internally. Add it as a direct dependency if application code imports its modules directly:

```toml
[dependencies]
TZoR = { path = "/path/to/TZoR" }
Utilities = { path = "/path/to/TZoR/Utilities" }
```

## Device descriptor

`Utilities::device::Device::Device` has two variants: `Cpu` and `Cuda(usize)`. Methods `is_cpu()` and `is_cuda()` check the variant. `Display` formats devices as `cpu` and `cuda:<ordinal>`. The TZoR crate exposes this type through `Tensor::device()` and accepts it in transfer methods. See [Storage](storage.md).

```rust
use Utilities::device::Device::Device;

let cpu = Device::Cpu;
assert!(cpu.is_cpu());
assert_eq!(cpu.to_string(), "cpu");
```

## Random number generator

`Utilities::random::Random::Rng` is a deterministic stateful generator. It uses a Xoshiro-family generator seeded from one `u64` and offers:

- `Rng::seed(seed)`: initialize generator state.
- `next_u64()`: next 64-bit value.
- `uniform()`: `f64` in `[0, 1)`.
- `normal()`: approximate standard normal `f64` generated with the Box–Muller transform.

```rust
use Utilities::random::Random::Rng;

let mut rng = Rng::seed(42);
let u = rng.uniform();
let z = rng.normal();
assert!((0.0..1.0).contains(&u));
println!("sample: {z}");
```

`Tensor<f32/f64>::rand(shape, &mut rng)` and `randn(shape, &mut rng)` use this generator and cast each sample into the tensor element type.

## Marker traits

`Utilities::misc` defines marker traits with no methods:

- `IntLike`: implemented for `i8`, `i16`, `i32`, `i64`, `isize`, `u8`, `u16`, `u32`, `u64`, and `usize`.
- `SignedIntLike`: implemented for signed integer types above and `f32`/`f64`.
- `FloatLike`: implemented for `f32` and `f64`.

These traits can be used as bounds in generic application code; the main tensor API uses its own `dtype::{_Element, Element, Float}` traits.

## Logger

`Utilities::logger` provides:

- `t_Logger::Frequency(n)`: configure a message-count flush threshold.
- `t_Logger::Size(n)`: configure a buffer-size threshold.
- `t_Logger::SizedFrequent(size, frequency)`: configure both thresholds.
- `LogType`: `Warn`, `Info`, `Error`, `Success`, `Debug`, `Trace`, and `Log` labels.
- `OutputStream`: `File(String)`, `Window(usize)`, and `FD(u16)` output selection.
- `Logger::new(mode, output, metadata)`: construct a logger.
- `Logger::LOG(level, message)`: format and buffer/write a log record, returning a boxed error on I/O failures.

The public logger struct contains `fflush`, `curf`, `sflush`, `buff`, `metadata`, `output_stream`, and `fmt` fields. If `fmt` is set, the implementation substitutes `{LOG_CATEGORY}`, `{LOG_LEVEL}`, `{MSSG}`, and `{TIMESTAMP}`. Otherwise it uses a default timestamped record.

Current implementation caveats: window output is unimplemented (`todo!()`), simultaneous size and frequency thresholds are unimplemented, and `flush` is private and empty. Treat this logger as an incomplete utility; avoid configurations or output paths that invoke unimplemented branches in production.

## Tensor utility contracts

`TZoR::Utilities::Utilities` defines generic traits intended to express a tensor-like interface. These are contracts only and do not have a concrete implementation wired to `Tensor<T>` in this repository:

- `Shape`: `shape()`, with default `ndim()` and `numel()`.
- `Storage`: associated `Elem`, plus `len()`, `device()`, `as_ptr()`, and `as_mut_ptr()`.
- `Layout`: `shape()`, `strides()`, `offset()`, `is_contiguous()`.
- `Tensor<T>`: composes `Shape`, `Layout`, `Storage`, arithmetic and the two contracts below; additionally declares `unpack`, `reshape`, `view`, and `On`.
- `MinMathsContract<M, T>`: scalar `Min`, `Max`, `Count`, `Mag`, `Percentile`, `Mean`, `Median`, `Mode`, `SD`, `Var`, `Cov`, and `Pearson_coeff`.
- `VectorOpsContract<M, T>`: `Dot`, `Cross`, `Norm`, `Mag`, `Empty`, `Eye`, `Arange`, and `Random`.
- `Quartile`: `Q1`, `Q2`, `Q3`, and `Q(u8)` variants.

These contract methods use their declared capitalization and should not be confused with the concrete `Tensor<T>` APIs, which use lowercase Rust method names and fallible results.

## Pipelines module

`Utilities::pipelines` is currently an empty public module. It has no public functions or types yet.
