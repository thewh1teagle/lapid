use super::*;

fn file(entries: Vec<SteerEntry>) -> SteerFile {
    SteerFile { name: "t".into(), architecture: "arch".into(), hidden_size: 4, entries, ..Default::default() }
}

fn entry(hook: Hook, layer: usize, op: SteerOp, scale: f32, vector: Vec<f32>) -> SteerEntry {
    SteerEntry { hook, layer, op, scale, vector }
}

fn eval(dev: &Device, t: &Tensor) -> Vec<f32> {
    let r = dev.eval(&[t]).expect("eval");
    dev.to_vec_f32(&r[0]).expect("read")
}

#[test]
fn spec_parse() {
    assert_eq!(SteerSpec::parse("a/b.safetensors").unwrap(), SteerSpec { path: "a/b.safetensors".into(), scale: 1.0 });
    assert_eq!(SteerSpec::parse("a/b.safetensors:0.5").unwrap(), SteerSpec { path: "a/b.safetensors".into(), scale: 0.5 });
    assert!(SteerSpec::parse("a/b.safetensors:x").is_err());
    assert_eq!(ranges(&[2, 3, 4, 7, 9, 10]), "2-4,7,9-10");
}

#[test]
fn save_open_check() {
    let path = std::env::temp_dir().join(format!("lapid-steer-{}.safetensors", std::process::id()));
    let f = file(vec![entry(Hook::Resid, 1, SteerOp::Add, 2.0, vec![1.0, 0.0, 0.0, 0.0]), entry(Hook::AttnOut, 3, SteerOp::Ablate, 2.5, vec![0.0, 3.0, 4.0, 0.0])]);
    f.save(&path).unwrap();
    let g = SteerFile::open(&path);
    let _ = std::fs::remove_file(&path);
    let g = g.unwrap();
    assert_eq!((g.name.as_str(), g.hidden_size, g.entries.len()), ("t", 4, 2));
    // Sorted by hook: attn_out first, normalized.
    let a = &g.entries[0];
    assert_eq!((a.hook, a.layer, a.op, a.scale), (Hook::AttnOut, 3, SteerOp::Ablate, 2.5));
    assert!((a.vector[1] - 0.6).abs() < 1e-6 && (a.vector[2] - 0.8).abs() < 1e-6);
    assert_eq!(g.entries[1].vector, vec![1.0, 0.0, 0.0, 0.0]);
    let all = [Hook::AttnOut, Hook::Resid];
    g.check("arch", 4, 4, &all).unwrap();
    assert_eq!(g.check("other", 4, 4, &all).unwrap_err().code, "steering");
    assert_eq!(g.check("arch", 8, 4, &all).unwrap_err().code, "steering");
    assert_eq!(g.check("arch", 4, 3, &all).unwrap_err().code, "steering");
    assert_eq!(g.check("arch", 4, 4, &[Hook::Resid]).unwrap_err().code, "unsupported");
    assert_eq!(g.describe()["entries"][0]["layers"], "3");
}

#[test]
fn apply_ops() {
    let dev = Device::cpu();
    let d = vec![0.0, 0.6, 0.8, 0.0];
    let f = file(vec![entry(Hook::AttnOut, 0, SteerOp::Ablate, 1.0, d.clone()), entry(Hook::Resid, 0, SteerOp::Add, 0.5, vec![1.0, 2.0, 3.0, 4.0])]);
    // Multiplier 0 (or no files): no steering at all.
    assert!(Steering::new(&dev, &[(&f, 0.0)]).unwrap().is_none());
    assert!(Steering::new(&dev, &[]).unwrap().is_none());
    let x_host = [1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 0.25, 2.0];
    let x = Tensor::from_f32(&x_host, &[2, 4]).unwrap();
    for mult in [1.0f32, 2.5] {
        let s = Steering::new(&dev, &[(&f, mult)]).unwrap().unwrap();
        assert_eq!(s.key(), format!("t*{mult}"));
        let y = eval(&dev, &s.apply(Hook::AttnOut, 0, &x).unwrap());
        for (r, row) in x_host.chunks(4).enumerate() {
            let p: f32 = row.iter().zip(&d).map(|(a, b)| a * b).sum();
            for c in 0..4 {
                let want = row[c] - mult * d[c] * p;
                assert!((y[r * 4 + c] - want).abs() < 1e-5, "{mult} {r} {c}: {} vs {want}", y[r * 4 + c]);
            }
        }
        let z = eval(&dev, &s.apply(Hook::Resid, 0, &x).unwrap());
        assert!((z[0] - (1.0 + 0.5 * mult)).abs() < 1e-6);
        // Untouched hooks / layers pass the tensor itself.
        assert_eq!(s.apply(Hook::FfnOut, 0, &x).unwrap().id(), x.id());
        assert_eq!(s.apply(Hook::AttnOut, 1, &x).unwrap().id(), x.id());
    }
    // Ablation with multiplier 1 removes the direction.
    let s = Steering::new(&dev, &[(&f, 1.0)]).unwrap().unwrap();
    let y = eval(&dev, &s.apply(Hook::AttnOut, 0, &x).unwrap());
    for row in y.chunks(4) {
        let p: f32 = row.iter().zip(&d).map(|(a, b)| a * b).sum();
        assert!(p.abs() < 1e-5);
    }
}
