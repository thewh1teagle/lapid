//! `lapid` — developer tooling (devices, model files, verification, benchmarks, kernels).
//! Apps (transcription, chat, ...) are lapid-hub library examples, not commands.
//! Routing only, except the kernel tooling (`kernels.rs`, `suite.rs`), which exists only
//! for this tool: every other command delegates to a library `cmd` function. Errors are printed as JSON (`{"error": code, "message": ...}`) to
//! stderr with exit code 1.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use lapid::{Error, Result};

mod kernels;
mod suite;
mod trace;

#[derive(Parser)]
#[command(name = "lapid", version, about = "lapid developer tooling: devices, model files, verification, benchmarks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List usable devices (CPU, Metal, CUDA and Vulkan).
    Devices,
    /// Convert a Hugging Face repo, local HF directory or ggml/GGUF file to .lapid.
    Convert {
        /// HF repo (openai/whisper-small), local directory, or ggml-*.bin / *.gguf.
        src: String,
        /// Weight dtype: fp32, fp16, int8, int4, q4_k, q5_k, q6_k or q4_k_m (passed to the converter).
        #[arg(long)]
        dtype: String,
        /// Output file (default: the lapid cache).
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Calibrated int4 (GPTQ on the model's own activations): `builtin` or a text file.
        /// LFM3 and EmbeddingGemma; needs --dtype int4.
        #[arg(long)]
        calibrate: Option<String>,
    },
    /// Print metadata and the tensor table of a .lapid file as JSON.
    Info {
        file: PathBuf,
        /// Also check every tensor's checksum.
        #[arg(long)]
        verify: bool,
    },
    /// Verify a model against reference activations (tools/reference/<model>_ref.py).
    Verify {
        /// Model family (see `lapid_hub::cmd::MODELS`), e.g. whisper.
        model: String,
        /// Reference directory with activations.safetensors and result.json
        /// (default: the model's, e.g. data/reference/whisper-small).
        #[arg(long)]
        reference: Option<PathBuf>,
        #[command(flatten)]
        model_args: ModelArgs,
    },
    /// Benchmark a model (`bench whisper data/jfk.wav`) or the kernel suite (`bench kernels`).
    Bench {
        /// Model family, or `kernels`.
        target: String,
        /// Model input (e.g. an audio file); the model picks a default when omitted.
        input: Option<PathBuf>,
        #[command(flatten)]
        model_args: ModelArgs,
        /// Measured runs (after one warmup run).
        #[arg(long, default_value_t = 3)]
        runs: usize,
        /// `bench kernels`: only cases whose name contains this.
        #[arg(long)]
        filter: Option<String>,
    },
    /// Serve models over an OpenAI-compatible HTTP API with a web UI (llama-server flags work too).
    Serve(lapid_server::cli::ServeArgs),
    /// Chat with a model in the terminal, or run it on inputs: WAV files to transcribe,
    /// text to speak or embed (llama-cli flags work too).
    Run(lapid_server::cli::RunArgs),
    /// Kernel tooling.
    Kernels {
        #[command(subcommand)]
        action: KernelsAction,
    },
}

#[derive(Subcommand)]
enum KernelsAction {
    /// Compile the kernel suite and write the generated kernels (Vulkan WGSL, CUDA PTX) and
    /// each case's schedule to DIR. For a model's own kernels run it with LAPID_DEBUG=1
    /// (kernels are written to $LAPID_KERNEL_DIR as they compile).
    Dump {
        dir: PathBuf,
        /// cpu, vulkan[:N], cuda[:N] or metal[:N] (default: best available).
        #[arg(long)]
        device: Option<String>,
        /// Only cases whose name contains this.
        #[arg(long)]
        filter: Option<String>,
    },
}

#[derive(Args)]
struct ModelArgs {
    /// HF repo, local HF directory or .lapid file (default: the model's reference checkpoint).
    /// Own id: `verify` / `bench` also have a positional `model`.
    #[arg(long = "model", id = "model_path", value_name = "MODEL")]
    model: Option<String>,
    /// Weight dtype: fp32, fp16, int8, int4, q4_k, q5_k, q6_k or q4_k_m (default: the model's).
    #[arg(long)]
    dtype: Option<String>,
    /// cpu, gpu, metal[:N], cuda[:N] or vulkan[:N] (default: best available, or $LAPID_DEVICE).
    #[arg(long)]
    device: Option<String>,
    /// Read weights in place from the mmap'd model file (default: only when it does not fit).
    #[arg(long, overrides_with = "no_mmap")]
    mmap: bool,
    /// Always copy weights to the device.
    #[arg(long = "no-mmap", overrides_with = "mmap")]
    no_mmap: bool,
}

impl ModelArgs {
    fn hub(self) -> lapid_hub::cmd::ModelArgs {
        if self.mmap {
            lapid::set_residency(lapid::Residency::Mmap);
        } else if self.no_mmap {
            lapid::set_residency(lapid::Residency::Upload);
        }
        lapid_hub::cmd::ModelArgs { model: self.model, dtype: self.dtype, device: self.device }
    }
}

fn pretty(v: &serde_json::Value) -> Result<String> {
    Ok(serde_json::to_string_pretty(v)?)
}

/// Runs the command; `Ok(false)` means "completed, but report a failure exit code".
fn run(cli: Cli) -> Result<bool> {
    match cli.command {
        Command::Devices => {
            for line in lapid::Device::list() {
                println!("{line}");
            }
            println!("best: {}", lapid::Device::best().description());
            Ok(true)
        }
        Command::Convert { src, dtype, out, calibrate } => {
            let path = lapid_hub::cmd::convert_with(&src, &dtype, out.as_deref(), calibrate.as_deref())?;
            println!("{}", path.display());
            Ok(true)
        }
        Command::Info { file, verify } => {
            let v = lapid_format::cmd::info(&file, verify)?;
            println!("{}", pretty(&v)?);
            Ok(true)
        }
        Command::Verify { model, reference, model_args } => {
            let v = lapid_hub::cmd::verify(&model, reference.as_deref(), &model_args.hub())?;
            println!("{}", pretty(&v)?);
            Ok(v.get("ok").and_then(serde_json::Value::as_bool).unwrap_or(false))
        }
        Command::Bench { target, input, model_args, runs, filter } => {
            let v = if target == "kernels" {
                if input.is_some() || model_args.model.is_some() || model_args.dtype.is_some() {
                    return Err(Error::new("invalid_argument", "bench kernels takes only --device, --runs and --filter"));
                }
                kernels::bench(model_args.device.as_deref(), filter.as_deref(), runs)?
            } else {
                lapid_hub::cmd::bench(&target, input.as_deref(), &model_args.hub(), runs)?
            };
            println!("{}", pretty(&v)?);
            Ok(true)
        }
        Command::Serve(args) => {
            lapid_server::cli::serve(args)?;
            Ok(true)
        }
        Command::Run(args) => {
            lapid_server::cli::run(args)?;
            Ok(true)
        }
        Command::Kernels { action: KernelsAction::Dump { dir, device, filter } } => {
            let v = kernels::dump(&dir, device.as_deref(), filter.as_deref())?;
            println!("{}", pretty(&v)?);
            Ok(true)
        }
    }
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse_from(lapid_server::cli::normalize_args(std::env::args())) {
        Ok(cli) => cli,
        Err(e) => {
            // --help / --version are not errors.
            if !e.use_stderr() {
                let _ = e.print();
                return ExitCode::SUCCESS;
            }
            let err = Error::new("invalid_argument", e.render().to_string().trim().to_string());
            eprintln!("{}", err.to_json());
            return ExitCode::from(1);
        }
    };
    let traced = trace::init();
    let result = run(cli);
    if traced {
        trace::report();
    }
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("{}", e.to_json());
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_is_valid_and_model_flag_does_not_clash() {
        Cli::command().debug_assert();
        let cli = Cli::try_parse_from(["lapid", "verify", "whisper", "--model", "x.lapid", "--dtype", "q4_k_m"]).expect("parse");
        let Command::Verify { model, model_args, .. } = cli.command else { panic!("not verify") };
        assert_eq!((model.as_str(), model_args.model.as_deref()), ("whisper", Some("x.lapid")));
        let cli = Cli::try_parse_from(["lapid", "run", "m.lapid", "--no-mmap", "-p", "hi"]).expect("parse");
        assert!(matches!(cli.command, Command::Run(a) if a.load.no_mmap && !a.load.mmap));
        let cli = Cli::try_parse_from(["lapid", "bench", "llama", "--mmap"]).expect("parse");
        assert!(matches!(cli.command, Command::Bench { model_args, .. } if model_args.mmap));
        let cli = Cli::try_parse_from(["lapid", "convert", "a/b", "--dtype", "q5_k"]).expect("parse");
        assert!(matches!(cli.command, Command::Convert { dtype, .. } if dtype == "q5_k"));
    }
}
