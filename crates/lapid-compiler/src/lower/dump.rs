//! Human/agent-readable schedule dump.
//!
//! ```text
//! #0 matmul f32 x f16^T -> f32 epi=add(x0,x1) batch=1 m=1500 n=768 k=768 a=@0/bs0/rs768 b=@0/bs0/rs768 extras=[@0(0,0,1)] in=[b0,b1,b2] out=b3
//! #1 elementwise [f32]->f16 expr=mul(x0,0.5) shape=[1152000] in=[b3@0(1)] out=b4@0(1)
//! buffers:
//!   b0 host 4608000 B
//!   b3 temp slot=0 4608000 B
//! slots: s0=4608000 B (total 4608000 B)
//! outputs:
//!   o0 = b4 f16 [1, 1500, 768]
//! ```

use std::fmt::{Debug, Write};

use crate::ir::{BufId, BufferSource, Expr, Kernel, Launch, MatView, Schedule, View};

pub(crate) fn dump(s: &Schedule) -> String {
    let mut o = String::new();
    for (i, l) in s.launches.iter().enumerate() {
        let _ = writeln!(o, "#{i} {}", launch_line(l));
    }
    let _ = writeln!(o, "buffers:");
    for (id, b) in s.buffers.iter().enumerate() {
        let src = match &b.source {
            BufferSource::Host(_) => "host".to_string(),
            BufferSource::Device(_) => "device".to_string(),
            BufferSource::Temp { slot } => format!("temp slot={slot}"),
            BufferSource::Output => "output".to_string(),
        };
        let _ = writeln!(o, "  b{id} {src} {} B", b.bytes);
    }
    let total: usize = s.slot_bytes.iter().sum();
    let slots: Vec<String> = s.slot_bytes.iter().enumerate().map(|(i, b)| format!("s{i}={b} B")).collect();
    let _ = writeln!(o, "slots: {} (total {total} B)", slots.join(" "));
    let _ = writeln!(o, "outputs:");
    for (i, out) in s.outputs.iter().enumerate() {
        let _ = writeln!(o, "  o{i} = b{} {} {:?}", out.buf, out.dtype.name(), out.shape);
    }
    o
}

fn lname<T: Debug>(x: &T) -> String {
    format!("{x:?}").to_lowercase()
}

/// `x{i}` = load i; constants printed as f32.
pub(crate) fn fmt_expr(e: &Expr) -> String {
    match e {
        Expr::Load(i) => format!("x{i}"),
        Expr::Const(bits) => format!("{}", Expr::const_value(*bits)),
        Expr::Unary(op, a) => format!("{}({})", lname(op), fmt_expr(a)),
        Expr::Binary(op, a, b) => format!("{}({},{})", lname(op), fmt_expr(a), fmt_expr(b)),
        Expr::Where(c, a, b) => format!("where({},{},{})", fmt_expr(c), fmt_expr(a), fmt_expr(b)),
        Expr::LoadBits(i) => format!("w{i}"),
        Expr::ConstBits(bits) => format!("{}i", *bits as i32),
        Expr::IntUnary(op, a) => format!("i{}({})", lname(op), fmt_expr(a)),
        Expr::IntBinary(op, a, b) => format!("i{}({},{})", lname(op), fmt_expr(a), fmt_expr(b)),
        Expr::Convert(c, a) => format!("{}({})", lname(c), fmt_expr(a)),
    }
}

fn fmt_view(v: &View) -> String {
    let st: Vec<String> = v.strides.iter().map(|s| s.to_string()).collect();
    format!("@{}({})", v.offset, st.join(","))
}

fn fmt_mat(v: &MatView) -> String {
    format!("@{}/bs{}/rs{}", v.offset, v.batch_stride, v.row_stride)
}

fn fmt_ins(ins: &[BufId], views: Option<&[View]>) -> String {
    let parts: Vec<String> = ins
        .iter()
        .enumerate()
        .map(|(i, b)| match views.and_then(|v| v.get(i)) {
            Some(v) => format!("b{b}{}", fmt_view(v)),
            None => format!("b{b}"),
        })
        .collect();
    format!("[{}]", parts.join(","))
}

fn dtypes(d: &[lapid_tensor::DType]) -> String {
    let n: Vec<&str> = d.iter().map(|x| x.name()).collect();
    format!("[{}]", n.join(","))
}

fn launch_line(l: &Launch) -> String {
    let out = l.output;
    match &l.kernel {
        Kernel::Elementwise(s, d) => format!(
            "elementwise {}->{} expr={} shape={:?} in={} out=b{out}{}",
            dtypes(&s.in_dtypes),
            s.out_dtype.name(),
            fmt_expr(&s.expr),
            d.shape,
            fmt_ins(&l.inputs, Some(d.in_views.as_slice())),
            fmt_view(&d.out_view)
        ),
        Kernel::Reduce(s, d) => format!(
            "reduce {} {}->{} expr={} shape={:?} in={} out=b{out}",
            lname(&s.op),
            dtypes(&s.in_dtypes),
            s.out_dtype.name(),
            fmt_expr(&s.expr),
            d.shape,
            fmt_ins(&l.inputs, Some(d.in_views.as_slice()))
        ),
        Kernel::MatMul(s, d) => {
            let extras: Vec<String> = d.extra_views.iter().map(fmt_view).collect();
            format!(
                "matmul {} x {}{} -> {} epi={} batch={} m={} n={} k={} a={} b={} extras=[{}] {} in={} out=b{out}{}",
                s.a_dtype.name(),
                s.b_dtype.name(),
                if s.b_transposed { "^T" } else { "" },
                s.out_dtype.name(),
                s.epilogue.as_ref().map_or_else(|| "none".to_string(), fmt_expr),
                d.batch,
                d.m,
                d.n,
                d.k,
                fmt_mat(&d.a),
                fmt_mat(&d.b),
                extras.join(","),
                dtypes(&s.extra_dtypes),
                fmt_ins(&l.inputs, None),
                if d.out_is_contiguous() { String::new() } else { fmt_mat(&d.out) }
            )
        }
        Kernel::Gather(s, d) => format!(
            "gather {}->{} rows={} cols={} count={} in={} out=b{out}",
            s.table_dtype.name(),
            s.out_dtype.name(),
            d.rows,
            d.cols,
            d.count,
            fmt_ins(&l.inputs, None)
        ),
        Kernel::Softmax(s, d) => format!(
            "softmax {}->{} rows={} cols={} in={} out=b{out}",
            s.in_dtype.name(),
            s.out_dtype.name(),
            d.rows,
            d.cols,
            fmt_ins(&l.inputs, None)
        ),
        Kernel::RmsNorm(s, d) => format!(
            "rms_norm {} params={} -> {} rows={} cols={} eps={} in={} out=b{out}",
            s.in_dtype.name(),
            s.param_dtype.name(),
            s.out_dtype.name(),
            d.rows,
            d.cols,
            f32::from_bits(d.eps),
            fmt_ins(&l.inputs, None)
        ),
        Kernel::LayerNorm(s, d) => format!(
            "layer_norm {} params={} -> {} rows={} cols={} eps={} in={} out=b{out}",
            s.in_dtype.name(),
            s.param_dtype.name(),
            s.out_dtype.name(),
            d.rows,
            d.cols,
            f32::from_bits(d.eps),
            fmt_ins(&l.inputs, None)
        ),
        Kernel::Attention(s, d) => {
            let mut views = vec![d.q.clone(), d.k.clone(), d.v.clone()];
            if let Some(m) = &d.mask {
                views.push(m.clone());
            }
            format!(
                "attention {} mask={} causal={} -> {} b={} h={} sq={} sk={} d={} scale={} in={} out=b{out}",
                s.dtype.name(),
                s.mask_dtype.map_or("none", |m| m.name()),
                s.causal,
                s.out_dtype.name(),
                d.batch,
                d.heads,
                d.sq,
                d.sk,
                d.d,
                f32::from_bits(d.scale),
                fmt_ins(&l.inputs, Some(views.as_slice()))
            )
        }
        Kernel::GatherAxis(s, d) => format!(
            "gather_axis {} outer={} n={} inner={} axis_len={} in={} out=b{out}",
            s.dtype.name(),
            d.outer,
            d.n,
            d.inner,
            d.axis_len,
            fmt_ins(&l.inputs, Some(&[d.x.clone(), d.idx.clone()]))
        ),
        Kernel::TopK(s, d) => format!(
            "topk {} rows={} cols={} k={} in={} out=b{out}",
            s.in_dtype.name(),
            d.rows,
            d.cols,
            d.k,
            fmt_ins(&l.inputs, None)
        ),
        Kernel::MatMulId(s, d) => format!(
            "matmul_id {} x {}^T -> {} t={} r={} e={} n={} k={} in={} out=b{out}",
            s.x_dtype.name(),
            s.w_dtype.name(),
            s.out_dtype.name(),
            d.t,
            d.r,
            d.e,
            d.n,
            d.k,
            fmt_ins(&l.inputs, None)
        ),
        Kernel::Sinkhorn(d) => format!(
            "sinkhorn count={} n={} iters={} eps={} in={} out=b{out}",
            d.count,
            d.n,
            d.iters,
            f32::from_bits(d.eps),
            fmt_ins(&l.inputs, None)
        ),
        Kernel::GatedDeltaNet(s, d) => format!(
            "gated_delta_net -> {} b={} t={} h={} dk={} dv={} scale={} in={} out=b{out}",
            s.out_dtype.name(),
            d.b,
            d.t,
            d.h,
            d.dk,
            d.dv,
            f32::from_bits(d.scale),
            fmt_ins(&l.inputs, None)
        ),
    }
}
