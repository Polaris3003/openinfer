//! Serializable report and metric types emitted by the benchmark runners.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RunInfo {
    pub(crate) command: &'static str,
    pub(crate) model_path: String,
    pub(crate) model_type: String,
    pub(crate) cuda_graph: bool,
    pub(crate) load_ms: f64,
    pub(crate) label: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PromptDescriptor {
    pub(crate) source: String,
    pub(crate) prompt_tokens: usize,
    pub(crate) prompt_preview: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DurationStats {
    pub(crate) avg_ms: f64,
    pub(crate) p50_ms: f64,
    pub(crate) p95_ms: f64,
    pub(crate) p99_ms: f64,
    pub(crate) max_ms: f64,
    pub(crate) samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CountStats {
    pub(crate) min: usize,
    pub(crate) max: usize,
    pub(crate) avg: f64,
    pub(crate) samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeneratedTokenTrace {
    pub(crate) hash: String,
    pub(crate) prefix: Vec<u32>,
    pub(crate) len: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RequestWorkload {
    pub(crate) prompt: PromptDescriptor,
    pub(crate) output_len: usize,
    pub(crate) concurrency: usize,
    pub(crate) warmup: usize,
    pub(crate) iters: usize,
    pub(crate) seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RequestMetrics {
    pub(crate) ttft_ms: DurationStats,
    pub(crate) first_decode_step_ms: Option<DurationStats>,
    pub(crate) steady_tpot_ms: Option<DurationStats>,
    pub(crate) e2e_ms: DurationStats,
    pub(crate) generated_tokens: CountStats,
    #[serde(default)]
    pub(crate) generated_token_traces: Vec<GeneratedTokenTrace>,
    pub(crate) request_tok_s: Option<f64>,
    pub(crate) decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RequestIterationTiming {
    pub(crate) index: usize,
    pub(crate) ttft_ms: f64,
    pub(crate) first_decode_step_ms: Option<f64>,
    pub(crate) steady_tpot_ms: Option<DurationStats>,
    pub(crate) e2e_ms: f64,
    pub(crate) generated_tokens: usize,
    pub(crate) generated_token_trace: GeneratedTokenTrace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SnapshotProfile {
    pub(crate) prompt_len: usize,
    pub(crate) output_len: usize,
    pub(crate) metrics: RequestMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SnapshotReport {
    pub(crate) commit: String,
    pub(crate) date: String,
    pub(crate) model: String,
    pub(crate) gpu: String,
    /// Parallel layout the snapshot was measured under (e.g. "tp1-dp8-deepep").
    /// Absent in snapshots that predate multi-GPU model lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) parallel: Option<String>,
    pub(crate) prefill_heavy: SnapshotProfile,
    pub(crate) decode_heavy: SnapshotProfile,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RequestReport {
    pub(crate) run: RunInfo,
    pub(crate) workload: RequestWorkload,
    pub(crate) metrics: RequestMetrics,
    pub(crate) iterations: Vec<RequestIterationTiming>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct MatrixWorkload {
    pub(crate) prompt_lens: Vec<usize>,
    pub(crate) output_lens: Vec<usize>,
    pub(crate) warmup: usize,
    pub(crate) iters: usize,
    pub(crate) seed: u64,
    pub(crate) synthetic_pattern: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct MatrixCell {
    pub(crate) prompt_len: usize,
    pub(crate) output_len: usize,
    pub(crate) ttft_ms: DurationStats,
    pub(crate) e2e_ms: DurationStats,
    pub(crate) first_decode_step_ms: Option<DurationStats>,
    pub(crate) steady_tpot_ms: Option<DurationStats>,
    pub(crate) generated_tokens: CountStats,
    pub(crate) request_tok_s: Option<f64>,
    pub(crate) decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct MatrixReport {
    pub(crate) run: RunInfo,
    pub(crate) workload: MatrixWorkload,
    pub(crate) cells: Vec<MatrixCell>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CurveWorkload {
    pub(crate) prompt: PromptDescriptor,
    pub(crate) output_len: usize,
    pub(crate) window: usize,
    pub(crate) warmup: usize,
    pub(crate) iters: usize,
    pub(crate) seed: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CurveWindow {
    pub(crate) ctx_start: usize,
    pub(crate) ctx_end: usize,
    pub(crate) tpot_ms: DurationStats,
    pub(crate) decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CurveReport {
    pub(crate) run: RunInfo,
    pub(crate) workload: CurveWorkload,
    pub(crate) windows: Vec<CurveWindow>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum BenchReport {
    Request(Box<RequestReport>),
    Matrix(MatrixReport),
    Curve(CurveReport),
}
