//! Tensor handles: creation from host data, constants, getters, refcounting, dtype helpers.

use std::ffi::{c_char, c_void};

use engine::{DType, Error, HostData, Op, Result, Tensor};

use crate::error::{LapidStatus, guard, invalid};
use crate::handle::{LapidTensor, put_tensor, put_value, release, retain, slice, str_arg, tensor};
use crate::{LapidDType, dtype_from_c, dtype_to_c};

/// Byte size of `shape` in `dtype`, validated so it never panics or overflows.
fn storage_bytes(shape: &[usize], dtype: DType) -> Result<usize> {
    if dtype.is_quant() {
        if shape.len() != 2 && shape.len() != 3 {
            return Err(Error::new("shape_mismatch", format!("quantized {} tensors must be 2-D or 3-D, got {shape:?}", dtype.name())));
        }
        let block = dtype.quant_format().map_or(DType::QUANT_GROUP, |f| f.block_elems());
        if !shape[shape.len() - 1].is_multiple_of(block) {
            return Err(Error::new("shape_mismatch", format!("quantized {} cols must be a multiple of {block}, got {shape:?}", dtype.name())));
        }
        // Rows * cols * 2 bounds every quantized layout (padding aside); reject overflow early.
        shape.iter().try_fold(2usize, |acc, &d| acc.checked_mul(d)).ok_or_else(|| invalid("tensor too large"))?;
        return Ok(dtype.storage_bytes(shape));
    }
    shape.iter().try_fold(dtype.size(), |acc, &d| acc.checked_mul(d)).ok_or_else(|| invalid("tensor too large"))
}

/// A host tensor holding a copy of the data at `data`: row-major, contiguous, in `dtype`
/// (for quantized dtypes, the lapid-quant storage layout). The byte count is implied by
/// `shape[0..rank]` and `dtype`; the caller may free `data` when this returns.
/// `rank == 0` makes a scalar (`shape` may then be NULL).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_tensor_from_data(
    data: *const c_void,
    shape: *const usize,
    rank: usize,
    dtype: LapidDType,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    guard(|| {
        let dtype = dtype_from_c(dtype)?;
        let shape = unsafe { slice(shape, rank, "shape")? };
        let nbytes = storage_bytes(shape, dtype)?;
        let bytes = unsafe { slice(data.cast::<u8>(), nbytes, "data")? }.to_vec();
        put_tensor(out, Tensor::from_host(HostData::from_vec(bytes), shape, dtype)?)
    })
}

/// A constant tensor: `value` broadcast to `shape` in `dtype` (no memory until used).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_tensor_full(
    value: f64,
    shape: *const usize,
    rank: usize,
    dtype: LapidDType,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    guard(|| {
        let dtype = dtype_from_c(dtype)?;
        if dtype.is_quant() {
            return Err(Error::new("unsupported_dtype", "constants cannot be quantized"));
        }
        let shape = unsafe { slice(shape, rank, "shape")? };
        put_tensor(out, Tensor::full(value, shape, dtype))
    })
}

/// `lapid_tensor_full` with value 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_tensor_zeros(
    shape: *const usize,
    rank: usize,
    dtype: LapidDType,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    unsafe { lapid_tensor_full(0.0, shape, rank, dtype, out) }
}

/// Add a reference to a tensor handle (the same pointer stays valid until every reference
/// is released with `lapid_tensor_free`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_tensor_retain(t: *mut LapidTensor) {
    unsafe { retain(t) }
}

/// Drop one reference. NULL is a no-op. Graph nodes built from this tensor keep what they
/// need alive, so inputs can be freed right after use.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_tensor_free(t: *mut LapidTensor) {
    unsafe { release(t) }
}

/// Number of dimensions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_tensor_rank(t: *const LapidTensor, out: *mut usize) -> LapidStatus {
    guard(|| put_value(out, unsafe { tensor(t)? }.rank()))
}

/// Copy the shape into `dims[0..cap]` and the rank into `*rank` (`rank` may be NULL).
/// `LAPID_ERR_INVALID_ARGUMENT` if `cap` is smaller than the rank (`*rank` is still set).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_tensor_shape(
    t: *const LapidTensor,
    dims: *mut usize,
    cap: usize,
    rank: *mut usize,
) -> LapidStatus {
    guard(|| {
        let t = unsafe { tensor(t)? };
        if !rank.is_null() {
            put_value(rank, t.rank())?;
        }
        if cap < t.rank() {
            return Err(invalid(format!("dims holds {cap} entries, tensor rank is {}", t.rank())));
        }
        if t.rank() > 0 {
            if dims.is_null() {
                return Err(invalid("dims is NULL"));
            }
            // SAFETY: `dims` has room for `cap >= rank` entries (caller contract).
            unsafe { std::ptr::copy_nonoverlapping(t.shape().as_ptr(), dims, t.rank()) };
        }
        Ok(())
    })
}

/// Element type code (`LAPID_DTYPE_*`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_tensor_dtype(t: *const LapidTensor, out: *mut LapidDType) -> LapidStatus {
    guard(|| put_value(out, dtype_to_c(unsafe { tensor(t)? }.dtype())))
}

/// Number of elements (product of the shape; 1 for scalars).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_tensor_numel(t: *const LapidTensor, out: *mut usize) -> LapidStatus {
    guard(|| put_value(out, unsafe { tensor(t)? }.numel()))
}

/// Storage size in bytes (quantized dtypes: the full block layout).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_tensor_nbytes(t: *const LapidTensor, out: *mut usize) -> LapidStatus {
    guard(|| {
        let t = unsafe { tensor(t)? };
        put_value(out, storage_bytes(t.shape(), t.dtype())?)
    })
}

/// True when the tensor is realized on a device (a result of `lapid_eval` / `lapid_upload`),
/// false for host data, constants and unevaluated graph nodes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_tensor_is_realized(t: *const LapidTensor, out: *mut bool) -> LapidStatus {
    guard(|| put_value(out, matches!(unsafe { tensor(t)? }.op(), Op::Device(_))))
}

/// The same node with a debug name attached (shows up in schedule dumps).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_tensor_named(
    t: *const LapidTensor,
    name: *const c_char,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    guard(|| {
        let t = unsafe { tensor(t)? };
        let name = unsafe { str_arg(name, "name")? };
        put_tensor(out, t.named(name))
    })
}

/// Static name of a dtype code ("f32", "f16", "bf16", "i32", "int8", "int4", "q4_k", "q5_k",
/// "q6_k"), or NULL for an unknown code. Never free it.
#[unsafe(no_mangle)]
pub extern "C" fn lapid_dtype_name(dtype: LapidDType) -> *const c_char {
    let s: &'static str = match dtype_from_c(dtype) {
        Ok(DType::F32) => "f32\0",
        Ok(DType::F16) => "f16\0",
        Ok(DType::BF16) => "bf16\0",
        Ok(DType::I32) => "i32\0",
        Ok(DType::Int8) => "int8\0",
        Ok(DType::Int4) => "int4\0",
        Ok(DType::Q4K) => "q4_k\0",
        Ok(DType::Q5K) => "q5_k\0",
        Ok(DType::Q6K) => "q6_k\0",
        Ok(DType::Mxfp4) => "mxfp4\0",
        Err(_) => return std::ptr::null(),
    };
    s.as_ptr().cast()
}

/// Parse a dtype name (as returned by `lapid_dtype_name`) into `*out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_dtype_from_name(name: *const c_char, out: *mut LapidDType) -> LapidStatus {
    guard(|| {
        let name = unsafe { str_arg(name, "dtype name")? };
        let d = DType::from_name(name).ok_or_else(|| invalid(format!("unknown dtype '{name}'")))?;
        put_value(out, dtype_to_c(d))
    })
}

/// Bytes needed to store a tensor of `shape[0..rank]` in `dtype` (for
/// `lapid_tensor_from_data` buffers; quantized dtypes include block padding).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_dtype_storage_bytes(
    dtype: LapidDType,
    shape: *const usize,
    rank: usize,
    out: *mut usize,
) -> LapidStatus {
    guard(|| {
        let dtype = dtype_from_c(dtype)?;
        let shape = unsafe { slice(shape, rank, "shape")? };
        put_value(out, storage_bytes(shape, dtype)?)
    })
}
