//! lapid-compiler: lowers a tensor graph to a `Schedule` of kernel launches.
//! `ir` and `backend` are CONTRACT modules (coordinator-owned). The lowering itself
//! (agent "compiler") lives in the other modules.

pub mod backend;
pub mod ir;

mod lower;

pub use backend::Backend;
pub use ir::*;

use lapid_tensor::{Result, Tensor};

#[derive(Clone, Debug)]
pub struct CompileOptions {
    /// Fuse elementwise chains into producers/consumers (matmul epilogues, reduce prologues).
    pub fuse: bool,
    /// Emit Softmax/LayerNorm/Attention kernels instead of decomposing them.
    pub native_composites: bool,
    /// Emit `Kernel::Sinkhorn` for matrices up to 8 x 8 (backends that have it, see
    /// `Backend::compile_options`) instead of its decomposition (~1 launch per step).
    pub native_sinkhorn: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions { fuse: true, native_composites: true, native_sinkhorn: false }
    }
}

/// Lower `outputs` to a schedule. Every output is materialized contiguous into an
/// `Output` buffer (in the same order as `outputs`), even if it is a leaf or a view.
pub fn compile(outputs: &[Tensor], opts: &CompileOptions) -> Result<Schedule> {
    lower::compile(outputs, opts)
}

/// Human/agent-readable dump of a schedule (one line per launch). Used by `lapid` debug
/// output and by tests.
pub fn dump(schedule: &Schedule) -> String {
    lower::dump(schedule)
}
