//! Schedules for graphs whose *dims* change from eval to eval (a decode step: KV length,
//! write position) without recompiling.
//!
//! The plan cache keys a graph by its structure (`GraphKey::words`) and keeps its integer
//! parameters (shapes, slice / write positions, leaf sizes) apart (`GraphKey::dims`). Within
//! one structure the compiler is piecewise affine in those dims, so two compiled samples
//! `(dims_a, plan_a)`, `(dims_b, plan_b)` with the same launch structure define a line: a
//! later graph whose dims lie on it, `dims = dims_a + t (dims_b - dims_a)`, gets
//! `num = num_a + t (num_b - num_a)` for every numeric field of the schedule (kernel dims,
//! views, buffer sizes, output shapes).
//!
//! Safety net: a line is only used after a third compile on it matched the prediction
//! exactly (`verified`), predictions must be integral, and a field may not move onto the
//! special values 0 / 1 that the compiler branches on (empty kernels, size-1 dims) unless
//! both samples had that value. Anything else compiles normally and restarts the line.

use std::sync::Arc;

use lapid_compiler::{Kernel, View};

use super::plan::{Plan, PlanSrc};

/// Visit every numeric (dims-dependent) field of `plan` in a fixed order. Spec fields,
/// f32-bit fields (scale, eps), buffer ids and slot numbers are structure, not visited.
fn walk(plan: &mut Plan, f: &mut dyn FnMut(&mut i64)) {
    // SAFETY: every pointer is a live, exclusively borrowed usize / isize field of `plan`
    // (8 bytes, same layout as i64; the bit casts are the plain `as` conversions).
    walk_ptrs(plan, &mut |p| f(unsafe { &mut *p }));
}

/// `walk` handing out raw pointers to the fields (see `Fields`).
fn walk_ptrs(plan: &mut Plan, f: &mut dyn FnMut(*mut i64)) {
    const _: () = assert!(size_of::<usize>() == 8 && size_of::<isize>() == 8);
    fn u(x: &mut usize, f: &mut dyn FnMut(*mut i64)) {
        f(x as *mut usize as *mut i64)
    }
    fn i(x: &mut isize, f: &mut dyn FnMut(*mut i64)) {
        f(x as *mut isize as *mut i64)
    }
    fn view(v: &mut View, f: &mut dyn FnMut(*mut i64)) {
        u(&mut v.offset, f);
        v.strides.iter_mut().for_each(|s| i(s, f));
    }
    for l in &mut plan.launches {
        match &mut l.kernel {
            Kernel::Elementwise(_, d) => {
                d.shape.iter_mut().for_each(|x| u(x, f));
                d.in_views.iter_mut().for_each(|v| view(v, f));
                view(&mut d.out_view, f);
            }
            Kernel::Reduce(_, d) => {
                d.shape.iter_mut().for_each(|x| u(x, f));
                d.in_views.iter_mut().for_each(|v| view(v, f));
            }
            Kernel::MatMul(_, d) => {
                for x in [&mut d.batch, &mut d.m, &mut d.n, &mut d.k] {
                    u(x, f);
                }
                for m in [&mut d.a, &mut d.b, &mut d.out] {
                    for x in [&mut m.offset, &mut m.batch_stride, &mut m.row_stride] {
                        u(x, f);
                    }
                }
                d.extra_views.iter_mut().for_each(|v| view(v, f));
            }
            Kernel::Gather(_, d) => [&mut d.rows, &mut d.cols, &mut d.count].into_iter().for_each(|x| u(x, f)),
            Kernel::Softmax(_, d) => [&mut d.rows, &mut d.cols].into_iter().for_each(|x| u(x, f)),
            Kernel::LayerNorm(_, d) | Kernel::RmsNorm(_, d) => [&mut d.rows, &mut d.cols].into_iter().for_each(|x| u(x, f)),
            Kernel::Attention(_, d) => {
                for x in [&mut d.batch, &mut d.heads, &mut d.sq, &mut d.sk, &mut d.d] {
                    u(x, f);
                }
                for v in [&mut d.q, &mut d.k, &mut d.v] {
                    view(v, f);
                }
                if let Some(m) = &mut d.mask {
                    view(m, f);
                }
            }
            Kernel::GatherAxis(_, d) => {
                for x in [&mut d.outer, &mut d.n, &mut d.inner, &mut d.axis_len] {
                    u(x, f);
                }
                view(&mut d.x, f);
                view(&mut d.idx, f);
            }
            Kernel::TopK(_, d) => [&mut d.rows, &mut d.cols, &mut d.k].into_iter().for_each(|x| u(x, f)),
            Kernel::MatMulId(_, d) => {
                [&mut d.t, &mut d.r, &mut d.e, &mut d.n, &mut d.k].into_iter().for_each(|x| u(x, f))
            }
            Kernel::GatedDeltaNet(_, d) => {
                [&mut d.b, &mut d.t, &mut d.h, &mut d.dk, &mut d.dv].into_iter().for_each(|x| u(x, f))
            }
            Kernel::Sinkhorn(d) => [&mut d.count, &mut d.n, &mut d.iters].into_iter().for_each(|x| u(x, f)),
        }
    }
    plan.bufs.iter_mut().for_each(|b| u(&mut b.bytes, f));
    for o in &mut plan.outputs {
        o.shape.iter_mut().for_each(|x| u(x, f));
    }
}

/// Numeric fields of `plan` (in `walk` order).
fn nums(plan: &Plan) -> Vec<i64> {
    let mut out = Vec::with_capacity(plan.launches.len() * 12);
    let mut p = plan.clone_shallow();
    walk(&mut p, &mut |x| out.push(*x));
    out
}

/// `plan` with every numeric field zeroed: equal skeletons = same launch structure.
fn skeleton(plan: &Plan) -> Plan {
    let mut p = plan.clone_shallow();
    walk(&mut p, &mut |x| *x = 0);
    p
}

fn same_skeleton(a: &Plan, b: &Plan) -> bool {
    a.launches.len() == b.launches.len()
        && a.bufs.len() == b.bufs.len()
        && a.outputs.len() == b.outputs.len()
        && a.slot_bytes.len() == b.slot_bytes.len()
        && a.launches.iter().zip(&b.launches).all(|(x, y)| x.inputs == y.inputs && x.output == y.output && x.kernel == y.kernel)
        && a.bufs.iter().zip(&b.bufs).all(|(x, y)| x.bytes == y.bytes && same_src(&x.src, &y.src))
        && a.outputs.iter().zip(&b.outputs).all(|(x, y)| x.buf == y.buf && x.dtype == y.dtype && x.shape == y.shape)
}

fn same_src(a: &PlanSrc, b: &PlanSrc) -> bool {
    match (a, b) {
        (PlanSrc::Leaf(x), PlanSrc::Leaf(y)) | (PlanSrc::Temp(x), PlanSrc::Temp(y)) => x == y,
        (PlanSrc::Output, PlanSrc::Output) => true,
        _ => false,
    }
}

/// A compiled sample of one graph structure.
#[derive(Clone)]
pub(crate) struct Sample {
    dims: Vec<u64>,
    nums: Vec<i64>,
    pub plan: Arc<Plan>,
}

impl Sample {
    pub(crate) fn new(dims: Vec<u64>, plan: Arc<Plan>) -> Sample {
        Sample { nums: nums(&plan), dims, plan }
    }
}

/// Compiled samples of one graph structure: one, or two with the same launch structure
/// (a line), `verified` once a third compile on the line matched the extrapolation.
pub(crate) enum Line {
    One(Sample),
    /// `scratch`: the last predicted plan with pointers to its moving fields, patched in
    /// place when nobody holds it any more. `moving`: the fields that differ between a and b.
    Two { a: Sample, b: Sample, verified: bool, scratch: Option<(Arc<Plan>, Fields)>, moving: Option<Moving> },
}

/// `t = num / den` such that `dims = a + t (b - a)` component-wise, if any.
fn param(a: &[u64], b: &[u64], dims: &[u64]) -> Option<(i128, i128)> {
    if a.len() != dims.len() || b.len() != dims.len() {
        return None;
    }
    let mut t: Option<(i128, i128)> = None;
    for ((&x, &y), &z) in a.iter().zip(b).zip(dims) {
        if x == y {
            // Most dims are fixed: plain compare, no wide math.
            if z != x {
                return None;
            }
            continue;
        }
        let (x, y, z) = (x as i128, y as i128, z as i128);
        let (d, n) = (y - x, z - x);
        match t {
            None => t = Some((n, d)),
            Some((tn, td)) => {
                if n * td != tn * d {
                    return None;
                }
            }
        }
    }
    t
}

/// `a + t (b - a)` for every field, or `None` if not integral or a field would move onto a
/// value the compiler treats specially (0 / 1).
fn extrapolate(a: &[i64], b: &[i64], (tn, td): (i128, i128)) -> Option<Vec<i64>> {
    let mut out = Vec::with_capacity(a.len());
    for (&x, &y) in a.iter().zip(b) {
        if x == y {
            out.push(x);
            continue;
        }
        let num = (y as i128 - x as i128) * tn;
        if num % td != 0 {
            return None;
        }
        let v = x as i128 + num / td;
        let lo = x.min(y) as i128;
        // Never cross 0 / 1 (and never go negative) when the samples stayed above them.
        if (lo >= 2 && v < 2) || (lo >= 1 && v < 1) || (lo >= 0 && v < 0) {
            return None;
        }
        out.push(i64::try_from(v).ok()?);
    }
    Some(out)
}

/// Set the numeric fields of `p` (same skeleton as the samples) to `vals` and recompute
/// the slot sizes from its temps.
fn assign(p: &mut Plan, vals: &[i64]) {
    let mut it = vals.iter();
    walk(p, &mut |x| {
        if let Some(&v) = it.next() {
            *x = v;
        }
    });
    let mut slots = vec![0usize; p.slot_bytes.len()];
    for b in &p.bufs {
        if let PlanSrc::Temp(s) = b.src
            && let Some(x) = slots.get_mut(s)
        {
            *x = (*x).max(b.bytes);
        }
    }
    p.slot_bytes = slots;
}

/// Indices of the numeric fields that differ between the two samples of a line.
pub(crate) struct Moving {
    idx: Vec<usize>,
}

impl Moving {
    fn new(a: &[i64], b: &[i64]) -> Moving {
        Moving { idx: a.iter().zip(b).enumerate().filter(|(_, (x, y))| x != y).map(|(i, _)| i).collect() }
    }
}

/// `extrapolate` restricted to the moving fields (the others equal both samples).
fn extrapolate_moving(a: &[i64], b: &[i64], idx: &[usize], t: (i128, i128)) -> Option<Vec<i64>> {
    let xs: Vec<i64> = idx.iter().map(|&i| a[i]).collect();
    let ys: Vec<i64> = idx.iter().map(|&i| b[i]).collect();
    extrapolate(&xs, &ys, t)
}

/// Raw pointers to the moving numeric fields of a plan (valid while the plan is neither
/// moved out of its `Arc` nor resized; `assign` and `set_fields` only overwrite values).
pub(crate) struct Fields(Vec<*mut i64>);

// SAFETY: the pointers target the plan stored next to them in the same `Line`, which is only
// accessed under the plan-cache lock.
unsafe impl Send for Fields {}

/// Pointers to the fields of `plan` at `idx` (walk order). Every numeric field is a
/// usize / isize, stored as 8 bytes.
fn field_ptrs(plan: &mut Plan, idx: &[usize]) -> Fields {
    let mut out = Vec::with_capacity(idx.len());
    let mut k = 0usize;
    let mut next = idx.iter().peekable();
    walk_ptrs(plan, &mut |p| {
        if next.peek() == Some(&&k) {
            out.push(p);
            next.next();
        }
        k += 1;
    });
    Fields(out)
}

/// Write `vals` through `fields` (of `p`, which the caller holds exclusively).
fn set_fields(p: &mut Plan, fields: &Fields, vals: &[i64]) {
    for (&f, &v) in fields.0.iter().zip(vals) {
        // SAFETY: `f` points into `p` (see `Fields`), which is borrowed mutably here.
        unsafe { *f = v };
    }
    let mut slots = vec![0usize; p.slot_bytes.len()];
    for b in &p.bufs {
        if let PlanSrc::Temp(s) = b.src
            && let Some(x) = slots.get_mut(s)
        {
            *x = (*x).max(b.bytes);
        }
    }
    p.slot_bytes = slots;
}

impl Line {
    /// Predicted plan for `dims`, if this line is verified and `dims` lies on it.
    pub(crate) fn predict(&mut self, dims: &[u64]) -> Option<Arc<Plan>> {
        let Line::Two { a, b, verified: true, scratch, moving } = self else { return None };
        let t = param(&a.dims, &b.dims, dims)?;
        let moving = moving.get_or_insert_with(|| Moving::new(&a.nums, &b.nums));
        let vals = extrapolate_moving(&a.nums, &b.nums, &moving.idx, t)?;
        if let Some((p, fields)) = scratch.as_mut().and_then(|(p, f)| Some((Arc::get_mut(p)?, f))) {
            // Only the moving fields differ from the last prediction (same skeleton).
            set_fields(p, fields, &vals);
        } else {
            let mut p = b.plan.clone_shallow();
            let mut all = nums(&p);
            for (&i, &v) in moving.idx.iter().zip(&vals) {
                all[i] = v;
            }
            assign(&mut p, &all);
            let fields = field_ptrs(&mut p, &moving.idx);
            *scratch = Some((Arc::new(p), fields));
            return scratch.as_ref().map(|(p, _)| p.clone());
        }
        scratch.as_ref().map(|(p, _)| p.clone())
    }

    /// Record a freshly compiled sample.
    pub(crate) fn record(line: Option<Line>, s: Sample) -> Line {
        let pair = |b: Sample, s: Sample| {
            if b.dims != s.dims && same_skeleton(&skeleton(&b.plan), &skeleton(&s.plan)) {
                Line::Two { a: b, b: s, verified: false, scratch: None, moving: None }
            } else {
                Line::One(s)
            }
        };
        match line {
            None => Line::One(s),
            Some(Line::One(b)) => pair(b, s),
            Some(Line::Two { a, b, .. }) => {
                // A further sample on the line verifies it when the prediction is exact.
                let exact = param(&a.dims, &b.dims, &s.dims)
                    .and_then(|t| extrapolate(&a.nums, &b.nums, t))
                    .is_some_and(|v| v == s.nums && same_skeleton(&skeleton(&b.plan), &skeleton(&s.plan)));
                if exact {
                    Line::Two { a, b, verified: true, scratch: None, moving: None }
                } else {
                    // Off the line (the direction changed) or a mismatch: restart from the
                    // last two samples.
                    pair(b, s)
                }
            }
        }
    }
}

/// Recent samples kept per structure to find more lines.
const HISTORY: usize = 4;
/// Lines kept per structure besides the main one.
const MAX_ALT: usize = 4;

/// Verified lines through non-consecutive samples, for dims that do not move on one line
/// from step to step but do every other step: e.g. the compressed rows of a ratio-2 KV
/// compressor grow on every second decode step (one line per parity), which restarts
/// the main line at every step.
#[derive(Default)]
pub(crate) struct AltLines {
    hist: std::collections::VecDeque<Sample>,
    lines: Vec<Line>,
}

impl AltLines {
    pub(crate) fn predict(&mut self, dims: &[u64]) -> Option<Arc<Plan>> {
        self.lines.iter_mut().find_map(|l| l.predict(dims))
    }

    /// Record a freshly compiled sample: when two recent samples `x`, `y` and `s` lie on
    /// one line and its extrapolation matches `s` exactly, `(y, s)` becomes a verified line.
    pub(crate) fn record(&mut self, s: &Sample) {
        let on_line = |x: &Sample, y: &Sample| {
            x.dims != y.dims
                && param(&x.dims, &y.dims, &s.dims).and_then(|t| extrapolate(&x.nums, &y.nums, t)).is_some_and(|v| v == s.nums)
                && same_skeleton(&skeleton(&x.plan), &skeleton(&s.plan))
                && same_skeleton(&skeleton(&y.plan), &skeleton(&s.plan))
        };
        let n = self.hist.len();
        let found = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).find(|&(i, j)| on_line(&self.hist[i], &self.hist[j]));
        if let Some((_, j)) = found {
            if self.lines.len() >= MAX_ALT {
                self.lines.remove(0);
            }
            self.lines.push(Line::Two { a: self.hist[j].clone(), b: s.clone(), verified: true, scratch: None, moving: None });
        }
        if self.hist.len() >= HISTORY {
            self.hist.pop_front();
        }
        self.hist.push_back(s.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::super::plan::{GraphKey, Lookup, Plan, PlanCache};
    use super::{extrapolate, nums, param, same_skeleton, skeleton};
    use crate::Device;
    use lapid_compiler::CompileOptions;
    use lapid_tensor::Tensor;
    use std::sync::Arc;

    /// A decode-like step: write row `n - 1` of a KV buffer in place, attend over `[0, n)`.
    fn step(cache: &Tensor, n: usize) -> Vec<Tensor> {
        let knew = Tensor::from_f32(&vec![0.5; 16], &[1, 1, 2, 8]).unwrap();
        let c2 = cache.write_slice(1, n - 1, &knew).unwrap();
        let kv = c2.slice(1, 0, n).unwrap().permute(&[0, 2, 1, 3]).unwrap();
        let q = Tensor::from_f32(&vec![0.25; 16], &[1, 2, 1, 8]).unwrap();
        let a = Tensor::attention(&q, &kv, &kv, None, 0.3, true).unwrap().mul_scalar(2.0).unwrap();
        vec![a, c2]
    }

    #[test]
    fn predicted_plans_equal_compiled_ones() {
        let dev = Device::cpu();
        let cache = dev.upload(&Tensor::zeros(&[1, 32, 2, 8], lapid_tensor::DType::F32).contiguous().unwrap()).unwrap();
        let mut pc = PlanCache::default();
        for n in 1..12 {
            let outs = step(&cache, n);
            let refs: Vec<&Tensor> = outs.iter().collect();
            let key = GraphKey::new(&refs);
            let compiled = Plan::new(lapid_compiler::compile(&outs, &CompileOptions::default()).unwrap(), &key);
            match pc.get(&key) {
                Some((p, how)) => {
                    assert_eq!(how, Lookup::Predicted, "n={n}");
                    assert!(n > 3, "predicted before the line was verified (n={n})");
                    assert_eq!(nums(&p), nums(&compiled), "n={n}");
                    assert!(same_skeleton(&skeleton(&p), &skeleton(&compiled)), "n={n}");
                    assert_eq!(p.slot_bytes, compiled.slot_bytes, "n={n}");
                }
                None => {
                    assert!(n <= 3, "no prediction at n={n}");
                    pc.insert(&key, Arc::new(compiled));
                }
            }
        }
    }

    /// Dims on two interleaved lines (a second KV of `(n + 1) / 2` rows, as a ratio-2
    /// compressor): predicted from the third sample of each parity on, always exact.
    #[test]
    fn every_other_step_lines() {
        let dev = Device::cpu();
        let cache = dev.upload(&Tensor::zeros(&[1, 32, 2, 8], lapid_tensor::DType::F32).contiguous().unwrap()).unwrap();
        let mut pc = PlanCache::default();
        let mut predicted = 0;
        for n in 1..24 {
            let mut outs = step(&cache, n);
            let c = n.div_ceil(2);
            let kv = cache.slice(1, 0, c).unwrap().permute(&[0, 2, 1, 3]).unwrap();
            let q = Tensor::from_f32(&vec![0.5; 16], &[1, 2, 1, 8]).unwrap();
            outs.push(Tensor::attention(&q, &kv, &kv, None, 0.3, false).unwrap());
            let refs: Vec<&Tensor> = outs.iter().collect();
            let key = GraphKey::new(&refs);
            let compiled = Plan::new(lapid_compiler::compile(&outs, &CompileOptions::default()).unwrap(), &key);
            match pc.get(&key) {
                Some((p, _)) => {
                    predicted += 1;
                    assert_eq!(nums(&p), nums(&compiled), "n={n}");
                    assert!(same_skeleton(&skeleton(&p), &skeleton(&compiled)), "n={n}");
                    assert_eq!(p.slot_bytes, compiled.slot_bytes, "n={n}");
                }
                None => pc.insert(&key, Arc::new(compiled)),
            }
        }
        assert!(predicted >= 14, "only {predicted} of 23 steps predicted");
    }

    #[test]
    fn line_parameter_and_extrapolation() {
        assert_eq!(param(&[5, 7, 1], &[6, 8, 1], &[9, 11, 1]), Some((4, 1)));
        assert_eq!(param(&[5, 7, 1], &[6, 8, 1], &[9, 10, 1]), None);
        assert_eq!(param(&[5, 7, 1], &[6, 8, 1], &[9, 11, 2]), None);
        assert_eq!(param(&[4, 4], &[6, 8], &[5, 6]), Some((1, 2)));
        assert_eq!(extrapolate(&[10, 3, 7], &[12, 3, 8], (3, 1)), Some(vec![16, 3, 10]));
        assert_eq!(extrapolate(&[10, 3, 7], &[12, 3, 8], (1, 2)), None, "7.5 is not integral");
        // Moving onto 1 from above is refused.
        assert_eq!(extrapolate(&[3], &[2], (2, 1)), None);
        assert_eq!(extrapolate(&[0], &[1], (5, 1)), Some(vec![5]));
    }
}
