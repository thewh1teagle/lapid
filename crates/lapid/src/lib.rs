//! lapid — fast inference on any device, in pure Rust.
//!
//! The torch level: lazy `Tensor`s built from ~25 primitives and a `Device` that compiles
//! and runs graphs on CPU or Vulkan. Model-building layers live in the `lapid-nn` crate.
//!
//! CONTRACT: every `pub` item in this file (and the signatures listed in
//! plans/lapid-v1/CONTRACT.md for `device`, `file`) is coordinator-owned.
//! Implementation: agent "runtime".

pub mod device;
pub mod file;
mod log;
pub mod residency;
mod runtime;

pub use device::{Device, DeviceInfo};
pub use file::LapidFile;
pub use log::{LogHook, set_log_hook};
pub use residency::{Advice, Residency, advise, populate, populate_all, prefetch, residency, set_read_ranges, set_residency};
pub use lapid_tensor::{
    BinaryOp, DType, DeviceBuffer, DeviceData, Error, HostData, Op, ReduceOp, Result, Tensor,
    UnaryOp, bail,
};

/// Re-exported for advanced users / tooling.
pub use lapid_compiler as compiler;
pub use lapid_format as format;
pub use lapid_quant as quant;
