use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use openinfer_qwen3::projection_report::ProjectionReportOptions;
use openinfer_qwen3::projection_report::generate_projection_report;

#[derive(Debug, Parser)]
#[command(
    about = "Compare real-weight Qwen3 split/fused projections for one TP rank",
    after_help = "Example:\n  cargo run --release -p openinfer-qwen3 --bin \
                  qwen3_projection_report -- --model-path models/Qwen3-4B \
                  --tp-size 2 --rank 0 --shapes 1,8,32,128,1024 --out report.json"
)]
struct Cli {
    #[arg(long, default_value = "models/Qwen3-4B")]
    model_path: String,
    #[arg(long, default_value_t = 1)]
    tp_size: usize,
    #[arg(long, default_value_t = 0)]
    rank: usize,
    /// CUDA device ordinal; defaults to the TP rank.
    #[arg(long)]
    device: Option<usize>,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1,2,4,8,16,32,64,128,512,1024,2048,4096,8192,10000"
    )]
    shapes: Vec<usize>,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 5)]
    iters: usize,
    #[arg(long)]
    out: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let report = generate_projection_report(&ProjectionReportOptions {
        model_path: cli.model_path,
        tp_size: cli.tp_size,
        rank: cli.rank,
        device_ordinal: cli.device.unwrap_or(cli.rank),
        shapes: cli.shapes,
        warmup: cli.warmup,
        iters: cli.iters,
    })?;
    if let Some(parent) = cli.out.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create report directory {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&cli.out, format!("{json}\n"))
        .with_context(|| format!("write {}", cli.out.display()))?;
    println!("{}", cli.out.display());
    Ok(())
}
