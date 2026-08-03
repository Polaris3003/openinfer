//! Real-weight numerical and aggregate timing report for split vs fused projections.
//!
//! This is deliberately model-local: it loads the exact rank-local Qwen3
//! weights, tunes small-N GEMMs with the same all-layer cold-weight rotation as
//! the executor, and launches both candidates on one CUDA stream. It is a
//! diagnostic report, not a replacement for the HF/LoRA correctness gates.

use std::cmp::Ordering;

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::sys;
use half::bf16;
use openinfer_core::ops;
use openinfer_core::tensor::DeviceContext;
use openinfer_core::tensor::HiddenStates;
use openinfer_kernels::ops::GEMM_LT_MAX_N;
use openinfer_kernels::ops::NumericPolicy;
use serde::Serialize;

use crate::DecodeOverlap;
use crate::Qwen3ProjectionFusionOptions;
use crate::config::TensorParallelConfig;
use crate::weights::ModelRuntimeConfig;
use crate::weights::Qwen3Model;

const QUANTILE_SAMPLE_LIMIT: usize = 65_536;

#[derive(Clone, Debug)]
pub struct ProjectionReportOptions {
    pub model_path: String,
    pub tp_size: usize,
    pub rank: usize,
    pub device_ordinal: usize,
    pub shapes: Vec<usize>,
    pub warmup: usize,
    pub iters: usize,
}

#[derive(Debug, Serialize)]
pub struct ProjectionReport {
    schema: u32,
    report_type: &'static str,
    model_path: String,
    tp_size: usize,
    rank: usize,
    device_ordinal: usize,
    numeric_policy: &'static str,
    tuning: &'static str,
    config: ProjectionModelConfig,
    cells: Vec<ProjectionShapeReport>,
}

#[derive(Debug, Serialize)]
struct ProjectionModelConfig {
    hidden_size: usize,
    local_q_dim: usize,
    local_kv_dim: usize,
    local_intermediate_size: usize,
    layers: usize,
    warmup: usize,
    iters: usize,
}

#[derive(Debug, Serialize)]
struct ProjectionShapeReport {
    phase: &'static str,
    tokens: usize,
    algorithms: ShapeAlgorithms,
    scratch: ScratchBytes,
    layers: Vec<ProjectionLayerReport>,
}

#[derive(Debug, Serialize)]
struct ShapeAlgorithms {
    backend: &'static str,
    split_q: Option<AlgorithmMetadata>,
    split_kv: Option<AlgorithmMetadata>,
    fused_qkv: Option<AlgorithmMetadata>,
    split_gate_up: Option<AlgorithmMetadata>,
    fused_gate_up: Option<AlgorithmMetadata>,
}

#[derive(Debug, Serialize)]
struct AlgorithmMetadata {
    algo_id: i32,
    tile_id: i32,
    stages_id: i32,
    splitk: i32,
    reduction_scheme: i32,
    swizzling: i32,
    custom_option: i32,
}

impl From<openinfer_kernels::ops::GemmLtAlgorithmMetadata> for AlgorithmMetadata {
    fn from(value: openinfer_kernels::ops::GemmLtAlgorithmMetadata) -> Self {
        Self {
            algo_id: value.algo_id,
            tile_id: value.tile_id,
            stages_id: value.stages_id,
            splitk: value.splitk,
            reduction_scheme: value.reduction_scheme,
            swizzling: value.swizzling,
            custom_option: value.custom_option,
        }
    }
}

#[derive(Debug, Serialize)]
struct ScratchBytes {
    qkv_split: usize,
    qkv_fused: usize,
    qkv_fused_extra: usize,
    gate_up_split: usize,
    gate_up_fused: usize,
    gate_up_fused_extra: isize,
}

#[derive(Debug, Serialize)]
struct ProjectionLayerReport {
    layer: usize,
    /// Three row-range GEMMs vs one full GEMM + bitwise QKV split.
    qkv: CandidateComparison,
    /// Two row-range GEMMs vs one full GEMM; delta is over raw gate/up values.
    gate_up: CandidateComparison,
    /// Projection + SiLU×mul aggregate; delta is over the activation output.
    swiglu: CandidateComparison,
}

#[derive(Debug, Serialize)]
struct CandidateComparison {
    split: LatencyStats,
    fused: LatencyStats,
    improvement_pct: f64,
    split_launches: usize,
    fused_launches: usize,
    delta: DeltaStats,
}

#[derive(Debug, Serialize)]
struct LatencyStats {
    avg_us: f64,
    p50_us: f64,
    p99_us: f64,
    min_us: f64,
    max_us: f64,
    samples: usize,
}

#[derive(Debug, Serialize)]
struct DeltaStats {
    elements: usize,
    exact_elements: usize,
    exact_fraction: f64,
    mean_abs: f64,
    p50_abs: f64,
    p99_abs: f64,
    max_abs: f64,
    nan_or_inf: usize,
    quantile_sample_size: usize,
    bf16_ulp_histogram: UlpHistogram,
}

#[derive(Debug, Default, Serialize)]
struct UlpHistogram {
    ulp_0: usize,
    ulp_1: usize,
    ulp_2: usize,
    ulp_3_4: usize,
    ulp_5_8: usize,
    ulp_9_16: usize,
    ulp_17_plus: usize,
}

struct DeltaAccumulator {
    total: usize,
    exact: usize,
    non_finite: usize,
    abs_sum: f64,
    max_abs: f32,
    quantile_stride: usize,
    quantile_sample: Vec<f32>,
    ulp: UlpHistogram,
}

impl DeltaAccumulator {
    fn new(total: usize) -> Self {
        Self {
            total,
            exact: 0,
            non_finite: 0,
            abs_sum: 0.0,
            max_abs: 0.0,
            quantile_stride: total.div_ceil(QUANTILE_SAMPLE_LIMIT).max(1),
            quantile_sample: Vec::with_capacity(total.min(QUANTILE_SAMPLE_LIMIT)),
            ulp: UlpHistogram::default(),
        }
    }

    fn observe(&mut self, index: usize, baseline: bf16, candidate: bf16) {
        let baseline_f32 = baseline.to_f32();
        let candidate_f32 = candidate.to_f32();
        if !baseline_f32.is_finite() || !candidate_f32.is_finite() {
            self.non_finite += 1;
            return;
        }
        if baseline.to_bits() == candidate.to_bits() {
            self.exact += 1;
        }
        let abs = (baseline_f32 - candidate_f32).abs();
        self.abs_sum += f64::from(abs);
        self.max_abs = self.max_abs.max(abs);
        if index % self.quantile_stride == 0 {
            self.quantile_sample.push(abs);
        }
        match bf16_ulp_distance(baseline, candidate) {
            0 => self.ulp.ulp_0 += 1,
            1 => self.ulp.ulp_1 += 1,
            2 => self.ulp.ulp_2 += 1,
            3..=4 => self.ulp.ulp_3_4 += 1,
            5..=8 => self.ulp.ulp_5_8 += 1,
            9..=16 => self.ulp.ulp_9_16 += 1,
            _ => self.ulp.ulp_17_plus += 1,
        }
    }

    fn finish(mut self) -> DeltaStats {
        self.quantile_sample
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let finite = self.total.saturating_sub(self.non_finite);
        DeltaStats {
            elements: self.total,
            exact_elements: self.exact,
            exact_fraction: ratio(self.exact, finite),
            mean_abs: if finite == 0 {
                f64::NAN
            } else {
                self.abs_sum / finite as f64
            },
            p50_abs: percentile_f32(&self.quantile_sample, 0.50),
            p99_abs: percentile_f32(&self.quantile_sample, 0.99),
            max_abs: f64::from(self.max_abs),
            nan_or_inf: self.non_finite,
            quantile_sample_size: self.quantile_sample.len(),
            bf16_ulp_histogram: self.ulp,
        }
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        f64::NAN
    } else {
        numerator as f64 / denominator as f64
    }
}

fn percentile_f32(sorted: &[f32], percentile: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    f64::from(sorted[index])
}

fn bf16_ordered(bits: u16) -> i32 {
    if bits & 0x8000 != 0 {
        0x8000 - i32::from(bits & 0x7fff)
    } else {
        0x8000 + i32::from(bits)
    }
}

fn bf16_ulp_distance(lhs: bf16, rhs: bf16) -> usize {
    bf16_ordered(lhs.to_bits())
        .abs_diff(bf16_ordered(rhs.to_bits()))
        .try_into()
        .unwrap_or(usize::MAX)
}

fn latency(samples: Vec<f64>) -> LatencyStats {
    let mut sorted = samples;
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let percentile = |p: f64| {
        let index = ((sorted.len() - 1) as f64 * p).round() as usize;
        sorted[index]
    };
    LatencyStats {
        avg_us: sorted.iter().sum::<f64>() / sorted.len() as f64,
        p50_us: percentile(0.50),
        p99_us: percentile(0.99),
        min_us: sorted[0],
        max_us: sorted[sorted.len() - 1],
        samples: sorted.len(),
    }
}

fn measure(
    ctx: &DeviceContext,
    warmup: usize,
    iters: usize,
    mut launch: impl FnMut() -> Result<()>,
) -> Result<LatencyStats> {
    for _ in 0..warmup {
        launch()?;
    }
    ctx.sync()?;
    let start = ctx
        .ctx
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = ctx
        .ctx
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        start.record(&ctx.stream)?;
        launch()?;
        end.record(&ctx.stream)?;
        samples.push(f64::from(start.elapsed_ms(&end)?) * 1_000.0);
    }
    ctx.sync()?;
    Ok(latency(samples))
}

fn hidden_to_bf16(ctx: &DeviceContext, hidden: &HiddenStates) -> Result<Vec<bf16>> {
    let host = ctx.stream.clone_dtoh(&hidden.data)?;
    ctx.sync()?;
    Ok(host)
}

fn patterned_input(ctx: &DeviceContext, hidden: usize, tokens: usize) -> Result<HiddenStates> {
    let host: Vec<bf16> = (0..hidden * tokens)
        .map(|index| {
            let mixed = (index as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let centered = ((mixed >> 48) as i32 - 32_768) as f32 / 32_768.0;
            bf16::from_f32(centered * 0.125)
        })
        .collect();
    HiddenStates::from_host(ctx, &host, hidden, tokens)
}

fn compare_direct(baseline: &[bf16], candidate: &[bf16]) -> DeltaStats {
    assert_eq!(baseline.len(), candidate.len());
    let mut stats = DeltaAccumulator::new(baseline.len());
    for (index, (&lhs, &rhs)) in baseline.iter().zip(candidate).enumerate() {
        stats.observe(index, lhs, rhs);
    }
    stats.finish()
}

fn compare_segmented(
    segments: &[(&[bf16], usize)],
    candidate: &[bf16],
    tokens: usize,
) -> DeltaStats {
    let combined_dim: usize = segments.iter().map(|(_, dim)| *dim).sum();
    assert_eq!(candidate.len(), combined_dim * tokens);
    for (segment, dim) in segments {
        assert_eq!(segment.len(), dim * tokens);
    }
    let mut stats = DeltaAccumulator::new(candidate.len());
    let mut index = 0;
    for token in 0..tokens {
        let mut candidate_offset = token * combined_dim;
        for (segment, dim) in segments {
            let segment_offset = token * dim;
            for element in 0..*dim {
                stats.observe(
                    index,
                    segment[segment_offset + element],
                    candidate[candidate_offset + element],
                );
                index += 1;
            }
            candidate_offset += dim;
        }
    }
    stats.finish()
}

fn comparison(
    split: LatencyStats,
    fused: LatencyStats,
    split_launches: usize,
    fused_launches: usize,
    delta: DeltaStats,
) -> CandidateComparison {
    let improvement_pct = (split.p50_us - fused.p50_us) / split.p50_us * 100.0;
    CandidateComparison {
        split,
        fused,
        improvement_pct,
        split_launches,
        fused_launches,
        delta,
    }
}

fn tune_shape(model: &Qwen3Model, tokens: usize) -> Result<()> {
    if tokens > GEMM_LT_MAX_N {
        return Ok(());
    }
    let q_dim = model.layers[0].attention.q_dim;
    let kv_dim = model.layers[0].attention.kv_dim;
    let qkv_dim = q_dim + 2 * kv_dim;
    let intermediate = model.layers[0].mlp.gate_up_proj.rows / 2;

    let q: Vec<_> = model
        .layers
        .iter()
        .map(|layer| (&layer.attention.qkv_proj, 0))
        .collect();
    let kv: Vec<_> = model
        .layers
        .iter()
        .flat_map(|layer| {
            [
                (&layer.attention.qkv_proj, q_dim),
                (&layer.attention.qkv_proj, q_dim + kv_dim),
            ]
        })
        .collect();
    let qkv: Vec<_> = model
        .layers
        .iter()
        .map(|layer| (&layer.attention.qkv_proj, 0))
        .collect();
    let gate_up_split: Vec<_> = model
        .layers
        .iter()
        .flat_map(|layer| {
            [
                (&layer.mlp.gate_up_proj, 0),
                (&layer.mlp.gate_up_proj, intermediate),
            ]
        })
        .collect();
    let gate_up_fused: Vec<_> = model
        .layers
        .iter()
        .map(|layer| (&layer.mlp.gate_up_proj, 0))
        .collect();
    ops::gemm_lt_tune(&model.ctx, &q, q_dim, tokens)?;
    ops::gemm_lt_tune(&model.ctx, &kv, kv_dim, tokens)?;
    ops::gemm_lt_tune(&model.ctx, &qkv, qkv_dim, tokens)?;
    ops::gemm_lt_tune(&model.ctx, &gate_up_split, intermediate, tokens)?;
    ops::gemm_lt_tune(&model.ctx, &gate_up_fused, 2 * intermediate, tokens)?;
    Ok(())
}

fn shape_algorithms(model: &Qwen3Model, tokens: usize) -> Result<ShapeAlgorithms> {
    if tokens > GEMM_LT_MAX_N {
        return Ok(ShapeAlgorithms {
            backend: "cublas_gemm_ex",
            split_q: None,
            split_kv: None,
            fused_qkv: None,
            split_gate_up: None,
            fused_gate_up: None,
        });
    }
    let hidden = model.config.hidden_size;
    let q_dim = model.layers[0].attention.q_dim;
    let kv_dim = model.layers[0].attention.kv_dim;
    let intermediate = model.layers[0].mlp.gate_up_proj.rows / 2;
    let query = |rows| -> Result<Option<AlgorithmMetadata>> {
        Ok(
            openinfer_kernels::ops::gemm_lt_algorithm_metadata(rows, tokens, hidden)?
                .map(Into::into),
        )
    };
    let algorithms = ShapeAlgorithms {
        backend: "cublas_lt_tuned",
        split_q: query(q_dim)?,
        split_kv: query(kv_dim)?,
        fused_qkv: query(q_dim + 2 * kv_dim)?,
        split_gate_up: query(intermediate)?,
        fused_gate_up: query(2 * intermediate)?,
    };
    ensure!(
        algorithms.split_q.is_some()
            && algorithms.split_kv.is_some()
            && algorithms.fused_qkv.is_some()
            && algorithms.split_gate_up.is_some()
            && algorithms.fused_gate_up.is_some(),
        "missing tuned cuBLASLt metadata for N={tokens}"
    );
    Ok(algorithms)
}

fn report_layer(
    model: &Qwen3Model,
    layer_index: usize,
    input: &HiddenStates,
    warmup: usize,
    iters: usize,
) -> Result<ProjectionLayerReport> {
    let ctx = &model.ctx;
    let layer = &model.layers[layer_index];
    let tokens = input.seq_len;
    let q_dim = layer.attention.q_dim;
    let kv_dim = layer.attention.kv_dim;
    let qkv_dim = q_dim + 2 * kv_dim;
    let intermediate = layer.mlp.gate_up_proj.rows / 2;

    let mut q = HiddenStates::zeros(ctx, q_dim, tokens)?;
    let mut k = HiddenStates::zeros(ctx, kv_dim, tokens)?;
    let mut v = HiddenStates::zeros(ctx, kv_dim, tokens)?;
    let mut qkv = HiddenStates::zeros(ctx, qkv_dim, tokens)?;
    let mut fused_q = HiddenStates::zeros(ctx, q_dim, tokens)?;
    let mut fused_k = HiddenStates::zeros(ctx, kv_dim, tokens)?;
    let mut fused_v = HiddenStates::zeros(ctx, kv_dim, tokens)?;

    let qkv_split_timing = measure(ctx, warmup, iters, || {
        ops::gemm_rows_into_checked(ctx, &layer.attention.qkv_proj, 0, q_dim, input, &mut q)?;
        ops::gemm_rows_into_checked(ctx, &layer.attention.qkv_proj, q_dim, kv_dim, input, &mut k)?;
        ops::gemm_rows_into_checked(
            ctx,
            &layer.attention.qkv_proj,
            q_dim + kv_dim,
            kv_dim,
            input,
            &mut v,
        )
    })?;
    let qkv_fused_timing = measure(ctx, warmup, iters, || {
        ops::gemm_into_checked(ctx, &layer.attention.qkv_proj, input, &mut qkv)?;
        openinfer_kernels::ops::split_qkv_into(ctx, &qkv, &mut fused_q, &mut fused_k, &mut fused_v)
    })?;
    let q_host = hidden_to_bf16(ctx, &q)?;
    let k_host = hidden_to_bf16(ctx, &k)?;
    let v_host = hidden_to_bf16(ctx, &v)?;
    let qkv_host = hidden_to_bf16(ctx, &qkv)?;
    let qkv_delta = compare_segmented(
        &[(&q_host, q_dim), (&k_host, kv_dim), (&v_host, kv_dim)],
        &qkv_host,
        tokens,
    );

    let mut gate = HiddenStates::zeros(ctx, intermediate, tokens)?;
    let mut up = HiddenStates::zeros(ctx, intermediate, tokens)?;
    let mut gate_up = HiddenStates::zeros(ctx, 2 * intermediate, tokens)?;
    let mut split_activation = HiddenStates::zeros(ctx, intermediate, tokens)?;
    let mut fused_activation = HiddenStates::zeros(ctx, intermediate, tokens)?;

    let gate_split_timing = measure(ctx, warmup, iters, || {
        ops::gemm_rows_into_checked(
            ctx,
            &layer.mlp.gate_up_proj,
            0,
            intermediate,
            input,
            &mut gate,
        )?;
        ops::gemm_rows_into_checked(
            ctx,
            &layer.mlp.gate_up_proj,
            intermediate,
            intermediate,
            input,
            &mut up,
        )
    })?;
    let gate_fused_timing = measure(ctx, warmup, iters, || {
        ops::gemm_into_checked(ctx, &layer.mlp.gate_up_proj, input, &mut gate_up)
    })?;
    let gate_host = hidden_to_bf16(ctx, &gate)?;
    let up_host = hidden_to_bf16(ctx, &up)?;
    let gate_up_host = hidden_to_bf16(ctx, &gate_up)?;
    let gate_up_delta = compare_segmented(
        &[(&gate_host, intermediate), (&up_host, intermediate)],
        &gate_up_host,
        tokens,
    );

    let swiglu_split_timing = measure(ctx, warmup, iters, || {
        ops::gemm_rows_into_checked(
            ctx,
            &layer.mlp.gate_up_proj,
            0,
            intermediate,
            input,
            &mut gate,
        )?;
        ops::gemm_rows_into_checked(
            ctx,
            &layer.mlp.gate_up_proj,
            intermediate,
            intermediate,
            input,
            &mut up,
        )?;
        ops::silu_mul_batch_into(ctx, &gate, &up, &mut split_activation)
    })?;
    let swiglu_fused_timing = measure(ctx, warmup, iters, || {
        ops::gemm_into_checked(ctx, &layer.mlp.gate_up_proj, input, &mut gate_up)?;
        ops::silu_mul_fused_batch_into(ctx, &gate_up, &mut fused_activation)
    })?;
    let split_activation_host = hidden_to_bf16(ctx, &split_activation)?;
    let fused_activation_host = hidden_to_bf16(ctx, &fused_activation)?;
    let swiglu_delta = compare_direct(&split_activation_host, &fused_activation_host);

    Ok(ProjectionLayerReport {
        layer: layer_index,
        qkv: comparison(qkv_split_timing, qkv_fused_timing, 3, 2, qkv_delta),
        gate_up: comparison(gate_split_timing, gate_fused_timing, 2, 1, gate_up_delta),
        swiglu: comparison(swiglu_split_timing, swiglu_fused_timing, 3, 2, swiglu_delta),
    })
}

pub fn generate_projection_report(options: &ProjectionReportOptions) -> Result<ProjectionReport> {
    ensure!(
        matches!(options.tp_size, 1 | 2),
        "projection report supports TP1/TP2, got TP{}",
        options.tp_size
    );
    ensure!(
        options.rank < options.tp_size,
        "rank {} must be below TP size {}",
        options.rank,
        options.tp_size
    );
    ensure!(!options.shapes.is_empty(), "at least one shape is required");
    ensure!(
        options.shapes.iter().all(|&shape| shape > 0),
        "all shapes must be > 0"
    );
    ensure!(options.warmup > 0, "warmup must be > 0");
    ensure!(options.iters > 0, "iters must be > 0");

    openinfer_kernels::ops::set_numeric_policy(NumericPolicy::Tuned);
    let model = Qwen3Model::from_safetensors_with_runtime(
        &options.model_path,
        ModelRuntimeConfig {
            enable_cuda_graph: false,
            tensor_parallel: Some(TensorParallelConfig {
                rank: options.rank,
                world_size: options.tp_size,
            }),
            device_ordinal: options.device_ordinal,
            max_loras: 1,
            max_lora_rank: 1,
            projection_fusion: Qwen3ProjectionFusionOptions::split(),
            decode_overlap: DecodeOverlap::Off,
        },
    )?;
    let q_dim = model.layers[0].attention.q_dim;
    let kv_dim = model.layers[0].attention.kv_dim;
    let intermediate = model.layers[0].mlp.gate_up_proj.rows / 2;
    let hidden = model.config.hidden_size;
    let mut cells = Vec::with_capacity(options.shapes.len());
    for &tokens in &options.shapes {
        tune_shape(&model, tokens)?;
        let algorithms = shape_algorithms(&model, tokens)?;
        let input = patterned_input(&model.ctx, hidden, tokens)?;
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer_index in 0..model.layers.len() {
            layers.push(report_layer(
                &model,
                layer_index,
                &input,
                options.warmup,
                options.iters,
            )?);
        }
        let bf16_bytes = std::mem::size_of::<bf16>();
        let qkv_split = (q_dim + 2 * kv_dim) * tokens * bf16_bytes;
        let qkv_fused = 2 * qkv_split;
        let gate_up_split = (3 * intermediate) * tokens * bf16_bytes;
        let gate_up_fused = (3 * intermediate) * tokens * bf16_bytes;
        cells.push(ProjectionShapeReport {
            phase: if tokens <= 64 { "decode" } else { "prefill" },
            tokens,
            algorithms,
            scratch: ScratchBytes {
                qkv_split,
                qkv_fused,
                qkv_fused_extra: qkv_fused - qkv_split,
                gate_up_split,
                gate_up_fused,
                gate_up_fused_extra: gate_up_fused as isize - gate_up_split as isize,
            },
            layers,
        });
    }
    Ok(ProjectionReport {
        schema: 1,
        report_type: "qwen3_projection_split_fused",
        model_path: options.model_path.clone(),
        tp_size: options.tp_size,
        rank: options.rank,
        device_ordinal: options.device_ordinal,
        numeric_policy: "tuned",
        tuning: "all-layer cold-weight rotation for N<=32; production default path for N>32",
        config: ProjectionModelConfig {
            hidden_size: hidden,
            local_q_dim: q_dim,
            local_kv_dim: kv_dim,
            local_intermediate_size: intermediate,
            layers: model.layers.len(),
            warmup: options.warmup,
            iters: options.iters,
        },
        cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_ulp_distance_handles_sign_and_adjacent_values() {
        assert_eq!(bf16_ulp_distance(bf16::ZERO, bf16::ZERO), 0);
        assert_eq!(
            bf16_ulp_distance(bf16::from_bits(0x3f80), bf16::from_bits(0x3f81)),
            1
        );
        assert_eq!(
            bf16_ulp_distance(bf16::from_bits(0xbf80), bf16::from_bits(0xbf81)),
            1
        );
        assert_eq!(bf16_ulp_distance(bf16::from_bits(0x8000), bf16::ZERO), 0);
    }

    #[test]
    fn latency_percentiles_are_stable() {
        let stats = latency(vec![4.0, 1.0, 3.0, 2.0]);
        assert_eq!(stats.p50_us, 3.0);
        assert_eq!(stats.p99_us, 4.0);
        assert_eq!(stats.avg_us, 2.5);
    }

    #[test]
    fn direct_delta_reports_exact_and_ulp_histogram() {
        let baseline = [bf16::from_bits(0x3f80), bf16::from_bits(0x4000)];
        let candidate = [bf16::from_bits(0x3f80), bf16::from_bits(0x4001)];
        let stats = compare_direct(&baseline, &candidate);
        assert_eq!(stats.exact_elements, 1);
        assert_eq!(stats.bf16_ulp_histogram.ulp_0, 1);
        assert_eq!(stats.bf16_ulp_histogram.ulp_1, 1);
    }
}
