//! Typed element access: dtype-dispatched loads into f32 chunks and stores from f32 chunks.
//!
//! All kernel arithmetic is f32. `Src` converts any plain storage dtype to f32 on load, `Dst`
//! converts f32 to the output dtype on store. Same-dtype copies (`copy_*`) move raw elements
//! so that i32 data survives bit-exactly.

use lapid_tensor::DType;
use half::slice::HalfFloatSliceExt;
use half::{bf16, f16};

/// IEEE half bits -> f32. Branch-free apart from inf/nan selection, so loops vectorize.
#[inline(always)]
pub(crate) fn f16_bits_to_f32(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let em = (h & 0x7fff) << 13;
    // Re-bias the exponent by multiplying with 2^112; handles subnormals exactly.
    let normal = (f32::from_bits(em) * f32::from_bits(0x7780_0000)).to_bits();
    let bits = if (h & 0x7c00) == 0x7c00 { em | 0x7f80_0000 } else { normal };
    f32::from_bits(bits | sign)
}

#[inline(always)]
pub(crate) fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[inline(always)]
fn f32_to_i32(v: f32) -> i32 {
    // Truncation toward zero, saturating; NaN -> 0.
    v as i32
}

/// `out[t] = f(s[base + t * stride])` without per-element bounds checks.
///
/// # Safety
/// The whole run must lie inside `s`.
#[inline(always)]
unsafe fn strided<T: Copy>(s: &[T], base: usize, stride: isize, out: &mut [f32], f: impl Fn(T) -> f32) {
    let mut p = s.as_ptr().wrapping_add(base);
    for o in out.iter_mut() {
        // SAFETY: in bounds per the caller.
        *o = f(unsafe { *p });
        p = p.wrapping_offset(stride);
    }
}

/// Read-only typed view of a buffer.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Src<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
    BF16(&'a [bf16]),
    I32(&'a [i32]),
}

/// Index of element `t` of a strided run starting at `base`.
#[inline(always)]
fn at(base: usize, stride: isize, t: usize) -> usize {
    (base as isize + t as isize * stride) as usize
}

/// Panics if the run `base + t*stride, t < n` leaves `[0, len)`. Views are validated before a
/// kernel starts, so this only fires on internal bugs; it keeps raw-pointer writes sound.
#[inline(always)]
fn check_run(base: usize, stride: isize, n: usize, len: usize) {
    if n == 0 {
        return;
    }
    let last = base as isize + (n as isize - 1) * stride;
    assert!(base < len && last >= 0 && (last as usize) < len, "cpu: element run out of bounds");
}

impl<'a> Src<'a> {
    pub fn len(&self) -> usize {
        match self {
            Src::F32(s) => s.len(),
            Src::F16(s) => s.len(),
            Src::BF16(s) => s.len(),
            Src::I32(s) => s.len(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Src::F32(_) => DType::F32,
            Src::F16(_) => DType::F16,
            Src::BF16(_) => DType::BF16,
            Src::I32(_) => DType::I32,
        }
    }

    pub fn as_f32(&self) -> Option<&'a [f32]> {
        match self {
            Src::F32(s) => Some(s),
            _ => None,
        }
    }

    #[inline]
    pub fn get(&self, i: usize) -> f32 {
        match self {
            Src::F32(s) => s[i],
            Src::F16(s) => f16_bits_to_f32(s[i].to_bits()),
            Src::BF16(s) => bf16_bits_to_f32(s[i].to_bits()),
            Src::I32(s) => s[i] as f32,
        }
    }

    /// `out[t] = self[base + t * stride]` converted to f32.
    pub fn load(&self, base: usize, stride: isize, out: &mut [f32]) {
        match stride {
            1 => self.load_contig(base, out),
            0 => {
                if !out.is_empty() {
                    let v = self.get(base);
                    out.fill(v);
                }
            }
            s => self.load_strided(base, s, out),
        }
    }

    pub fn load_contig(&self, base: usize, out: &mut [f32]) {
        let n = out.len();
        match self {
            Src::F32(s) => out.copy_from_slice(&s[base..base + n]),
            Src::F16(s) => s[base..base + n].convert_to_f32_slice(out),
            Src::BF16(s) => {
                for (o, v) in out.iter_mut().zip(&s[base..base + n]) {
                    *o = bf16_bits_to_f32(v.to_bits());
                }
            }
            Src::I32(s) => {
                for (o, v) in out.iter_mut().zip(&s[base..base + n]) {
                    *o = *v as f32;
                }
            }
        }
    }

    fn load_strided(&self, base: usize, stride: isize, out: &mut [f32]) {
        check_run(base, stride, out.len(), self.len());
        // SAFETY: the run is in bounds (checked above).
        unsafe {
            match self {
                Src::F32(s) => strided(s, base, stride, out, |v| v),
                Src::F16(s) => strided(s, base, stride, out, |v| f16_bits_to_f32(v.to_bits())),
                Src::BF16(s) => strided(s, base, stride, out, |v| bf16_bits_to_f32(v.to_bits())),
                Src::I32(s) => strided(s, base, stride, out, |v| v as f32),
            }
        }
    }

    /// `out[t] = self[offs[t]]`.
    pub fn gather(&self, offs: &[usize], out: &mut [f32]) {
        match self {
            Src::F32(s) => {
                for (o, &i) in out.iter_mut().zip(offs) {
                    *o = s[i];
                }
            }
            Src::F16(s) => {
                for (o, &i) in out.iter_mut().zip(offs) {
                    *o = f16_bits_to_f32(s[i].to_bits());
                }
            }
            Src::BF16(s) => {
                for (o, &i) in out.iter_mut().zip(offs) {
                    *o = bf16_bits_to_f32(s[i].to_bits());
                }
            }
            Src::I32(s) => {
                for (o, &i) in out.iter_mut().zip(offs) {
                    *o = s[i] as f32;
                }
            }
        }
    }
}

/// Raw, shareable output handle. Writes go through raw pointers; see `buffer.rs` for the
/// invariants that make concurrent use sound.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dst {
    ptr: *mut u8,
    len: usize,
    dtype: DType,
}

// SAFETY: tasks only write disjoint element ranges (buffer.rs invariant 3).
unsafe impl Send for Dst {}
unsafe impl Sync for Dst {}

impl Dst {
    pub(crate) fn new(ptr: *mut u8, len: usize, dtype: DType) -> Dst {
        Dst { ptr, len, dtype }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Base pointer as f32 (callers check the dtype and bounds themselves).
    pub(crate) fn f32_ptr(&self) -> *mut f32 {
        self.ptr as *mut f32
    }

    /// Mutable f32 slice of the output (dtype must be F32).
    ///
    /// # Safety
    /// No other live reference may overlap `[base, base + n)`.
    // Interior mutability through the buffer's raw pointer (threads write disjoint ranges).
    #[allow(clippy::mut_from_ref, dead_code)]
    pub unsafe fn f32_mut(&self, base: usize, n: usize) -> &mut [f32] {
        assert!(self.dtype == DType::F32 && base + n <= self.len, "cpu: bad f32 output slice");
        // SAFETY: in bounds (asserted), aligned (buffer is 64-aligned, offsets in elements).
        unsafe { std::slice::from_raw_parts_mut((self.ptr as *mut f32).add(base), n) }
    }

    /// `self[base + t * stride] = src[t]` with conversion to the output dtype.
    ///
    /// # Safety
    /// No other task may write the same elements concurrently.
    pub unsafe fn store(&self, base: usize, stride: isize, src: &[f32]) {
        let n = src.len();
        if n == 0 {
            return;
        }
        check_run(base, stride, n, self.len);
        // SAFETY: run checked in bounds; disjointness guaranteed by the caller.
        unsafe {
            if stride == 1 {
                match self.dtype {
                    DType::F32 => std::slice::from_raw_parts_mut((self.ptr as *mut f32).add(base), n)
                        .copy_from_slice(src),
                    DType::F16 => std::slice::from_raw_parts_mut((self.ptr as *mut f16).add(base), n)
                        .convert_from_f32_slice(src),
                    DType::BF16 => {
                        let d = std::slice::from_raw_parts_mut((self.ptr as *mut bf16).add(base), n);
                        for (o, &v) in d.iter_mut().zip(src) {
                            *o = bf16::from_f32(v);
                        }
                    }
                    DType::I32 => {
                        let d = std::slice::from_raw_parts_mut((self.ptr as *mut i32).add(base), n);
                        for (o, &v) in d.iter_mut().zip(src) {
                            *o = f32_to_i32(v);
                        }
                    }
                    DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K | DType::Mxfp4 => unreachable!("Dst never holds quantized dtypes"),
                }
                return;
            }
            match self.dtype {
                DType::F32 => {
                    let p = self.ptr as *mut f32;
                    for (t, &v) in src.iter().enumerate() {
                        *p.add(at(base, stride, t)) = v;
                    }
                }
                DType::F16 => {
                    let p = self.ptr as *mut f16;
                    for (t, &v) in src.iter().enumerate() {
                        *p.add(at(base, stride, t)) = f16::from_f32(v);
                    }
                }
                DType::BF16 => {
                    let p = self.ptr as *mut bf16;
                    for (t, &v) in src.iter().enumerate() {
                        *p.add(at(base, stride, t)) = bf16::from_f32(v);
                    }
                }
                DType::I32 => {
                    let p = self.ptr as *mut i32;
                    for (t, &v) in src.iter().enumerate() {
                        *p.add(at(base, stride, t)) = f32_to_i32(v);
                    }
                }
                DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K | DType::Mxfp4 => unreachable!("Dst never holds quantized dtypes"),
            }
        }
    }

    /// `self[offs[t]] = src[t]`.
    ///
    /// # Safety
    /// No other task may write the same elements concurrently.
    pub unsafe fn scatter(&self, offs: &[usize], src: &[f32]) {
        for &o in offs {
            assert!(o < self.len, "cpu: scatter out of bounds");
        }
        // SAFETY: every offset checked in bounds above.
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let p = self.ptr as *mut f32;
                    for (&o, &v) in offs.iter().zip(src) {
                        *p.add(o) = v;
                    }
                }
                DType::F16 => {
                    let p = self.ptr as *mut f16;
                    for (&o, &v) in offs.iter().zip(src) {
                        *p.add(o) = f16::from_f32(v);
                    }
                }
                DType::BF16 => {
                    let p = self.ptr as *mut bf16;
                    for (&o, &v) in offs.iter().zip(src) {
                        *p.add(o) = bf16::from_f32(v);
                    }
                }
                DType::I32 => {
                    let p = self.ptr as *mut i32;
                    for (&o, &v) in offs.iter().zip(src) {
                        *p.add(o) = f32_to_i32(v);
                    }
                }
                DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K | DType::Mxfp4 => unreachable!("Dst never holds quantized dtypes"),
            }
        }
    }
}

/// Raw element copy between equal dtypes: `dst[db + t*ds] = src[sb + t*ss]`.
/// Returns false (and does nothing) if dtypes differ.
///
/// # Safety
/// No other task may write the destination elements concurrently.
pub(crate) unsafe fn copy_run(src: &Src, sb: usize, ss: isize, dst: &Dst, db: usize, ds: isize, n: usize) -> bool {
    if src.dtype() != dst.dtype {
        return false;
    }
    check_run(sb, ss, n, src.len());
    check_run(db, ds, n, dst.len);
    // SAFETY: both runs checked in bounds; src and dst are distinct buffers.
    unsafe {
        match src {
            Src::F32(s) => copy_t(s, sb, ss, dst.ptr as *mut f32, db, ds, n),
            Src::I32(s) => copy_t(s, sb, ss, dst.ptr as *mut i32, db, ds, n),
            Src::F16(s) => copy_t(s, sb, ss, dst.ptr as *mut f16, db, ds, n),
            Src::BF16(s) => copy_t(s, sb, ss, dst.ptr as *mut bf16, db, ds, n),
        }
    }
    true
}

unsafe fn copy_t<T: Copy>(s: &[T], sb: usize, ss: isize, d: *mut T, db: usize, ds: isize, n: usize) {
    // SAFETY: bounds checked by the caller.
    unsafe {
        if ss == 1 && ds == 1 {
            std::ptr::copy_nonoverlapping(s.as_ptr().add(sb), d.add(db), n);
        } else if ss == 0 {
            let v = s[sb];
            for t in 0..n {
                *d.add(at(db, ds, t)) = v;
            }
        } else {
            let (mut sp, mut dp) = (s.as_ptr().wrapping_add(sb), d.wrapping_add(db));
            for _ in 0..n {
                *dp = *sp;
                sp = sp.wrapping_offset(ss);
                dp = dp.wrapping_offset(ds);
            }
        }
    }
}

/// Raw element copy with explicit offsets (equal dtypes). Returns false if dtypes differ.
///
/// # Safety
/// No other task may write the destination elements concurrently.
pub(crate) unsafe fn copy_offsets(src: &Src, soffs: &[usize], dst: &Dst, doffs: &[usize]) -> bool {
    if src.dtype() != dst.dtype {
        return false;
    }
    for &o in doffs {
        assert!(o < dst.len, "cpu: copy out of bounds");
    }
    // SAFETY: destination offsets checked; source reads are bounds-checked slice indexing.
    unsafe {
        match src {
            Src::F32(s) => gather_t(s, soffs, dst.ptr as *mut f32, doffs),
            Src::I32(s) => gather_t(s, soffs, dst.ptr as *mut i32, doffs),
            Src::F16(s) => gather_t(s, soffs, dst.ptr as *mut f16, doffs),
            Src::BF16(s) => gather_t(s, soffs, dst.ptr as *mut bf16, doffs),
        }
    }
    true
}

unsafe fn gather_t<T: Copy>(s: &[T], soffs: &[usize], d: *mut T, doffs: &[usize]) {
    for (&so, &dof) in soffs.iter().zip(doffs) {
        // SAFETY: dof checked by the caller.
        unsafe { *d.add(dof) = s[so] };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_matches_half() {
        for bits in (0u32..=0xffff).step_by(7) {
            let h = f16::from_bits(bits as u16);
            let a = f16_bits_to_f32(bits as u16);
            let b = h.to_f32();
            assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "{bits:#x}: {a} vs {b}");
        }
    }

    #[test]
    fn strided_load_store() {
        let v: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let s = Src::F32(&v);
        let mut out = [0f32; 4];
        s.load(18, -5, &mut out);
        assert_eq!(out, [18.0, 13.0, 8.0, 3.0]);
        s.load(7, 0, &mut out);
        assert_eq!(out, [7.0; 4]);
        let mut o = vec![f16::ZERO; 10];
        let d = Dst::new(o.as_mut_ptr() as *mut u8, 10, DType::F16);
        unsafe { d.store(1, 3, &[1.0, 2.0, 3.0]) };
        assert_eq!(o[1].to_f32(), 1.0);
        assert_eq!(o[4].to_f32(), 2.0);
        assert_eq!(o[7].to_f32(), 3.0);
    }
}
