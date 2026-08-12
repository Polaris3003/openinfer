use anyhow::Result;
use pegainfer_kernels::ops::NumericPolicy;
use serde::Serialize;

use crate::DecodeOverlap;
use crate::config::Config;
use crate::config::TensorParallelConfig;

/// User-facing control for one Qwen3 projection family.
///
/// `Auto` is deliberately conservative: only combinations present in the
/// measured production whitelist are fused. `ForceFused` is a diagnostic/A-B
/// control and errors at construction for an unsupported environment instead
/// of silently falling back.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProjectionFusionControl {
    #[default]
    Auto,
    Split,
    ForceFused,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen3ProjectionFusionOptions {
    pub qkv: ProjectionFusionControl,
    pub gate_up: ProjectionFusionControl,
}

impl Qwen3ProjectionFusionOptions {
    pub const fn split() -> Self {
        Self {
            qkv: ProjectionFusionControl::Split,
            gate_up: ProjectionFusionControl::Split,
        }
    }

    pub const fn force_fused() -> Self {
        Self {
            qkv: ProjectionFusionControl::ForceFused,
            gate_up: ProjectionFusionControl::ForceFused,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProjectionPhase {
    Decode,
    PrefillUnified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionFusionResolutionReason {
    ExplicitSplit,
    ForcedFused,
    AutoWhitelisted,
    AutoNotWhitelisted,
    UnsupportedGeometry,
    UnsupportedTensorParallel,
    UnsupportedNumericPolicy,
    UnsupportedDecodeOverlap,
    EmptyLocalProjection,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ProjectionFusionDecision {
    pub fused: bool,
    pub reason: ProjectionFusionResolutionReason,
}

impl ProjectionFusionDecision {
    const fn new(fused: bool, reason: ProjectionFusionResolutionReason) -> Self {
        Self { fused, reason }
    }
}

/// Construction-time Qwen3 projection topology.
///
/// The same value controls scratch allocation, GEMM tuning and CUDA Graph
/// capture. It is serializable so benchmark artifacts can record the actual
/// topology instead of only the requested CLI mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Qwen3ProjectionFusionPlan {
    pub decode_qkv: ProjectionFusionDecision,
    pub decode_gate_up: ProjectionFusionDecision,
    pub prefill_unified_qkv: ProjectionFusionDecision,
    pub prefill_unified_gate_up: ProjectionFusionDecision,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProjectionFusionEnvironment {
    pub(crate) numeric_policy: NumericPolicy,
    pub(crate) decode_overlap: DecodeOverlap,
}

#[derive(Clone, Copy, Debug)]
enum ProjectionKind {
    Qkv,
    GateUp,
}

impl ProjectionKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Qkv => "qkv",
            Self::GateUp => "gate/up",
        }
    }
}

impl Qwen3ProjectionFusionPlan {
    pub(crate) fn resolve(
        options: Qwen3ProjectionFusionOptions,
        config: &Config,
        tensor_parallel: TensorParallelConfig,
        environment: ProjectionFusionEnvironment,
    ) -> Result<Self> {
        let decode_qkv = resolve_one(
            options.qkv,
            ProjectionKind::Qkv,
            ProjectionPhase::Decode,
            config,
            tensor_parallel,
            environment,
        )?;
        let decode_gate_up = resolve_one(
            options.gate_up,
            ProjectionKind::GateUp,
            ProjectionPhase::Decode,
            config,
            tensor_parallel,
            environment,
        )?;
        let prefill_unified_qkv = resolve_one(
            options.qkv,
            ProjectionKind::Qkv,
            ProjectionPhase::PrefillUnified,
            config,
            tensor_parallel,
            environment,
        )?;
        let prefill_unified_gate_up = resolve_one(
            options.gate_up,
            ProjectionKind::GateUp,
            ProjectionPhase::PrefillUnified,
            config,
            tensor_parallel,
            environment,
        )?;
        Ok(Self {
            decode_qkv,
            decode_gate_up,
            prefill_unified_qkv,
            prefill_unified_gate_up,
        })
    }

    pub(crate) const fn qkv(self, phase: ProjectionPhase) -> bool {
        match phase {
            ProjectionPhase::Decode => self.decode_qkv.fused,
            ProjectionPhase::PrefillUnified => self.prefill_unified_qkv.fused,
        }
    }

    pub(crate) const fn gate_up(self, phase: ProjectionPhase) -> bool {
        match phase {
            ProjectionPhase::Decode => self.decode_gate_up.fused,
            ProjectionPhase::PrefillUnified => self.prefill_unified_gate_up.fused,
        }
    }
}

/// Resolve the exact construction-time projection plan without loading model
/// weights. Benchmarks call this before engine launch and persist the result;
/// model loading calls the same internal resolver again before allocating
/// scratch, so a successful run cannot report a different topology.
pub fn resolve_projection_fusion_plan(
    model_path: &str,
    tp_size: usize,
    options: Qwen3ProjectionFusionOptions,
    decode_overlap: DecodeOverlap,
) -> Result<Qwen3ProjectionFusionPlan> {
    anyhow::ensure!(tp_size > 0, "Qwen3 projection fusion requires tp_size > 0");
    let config = Config::from_file(model_path)?;
    let tensor_parallel = TensorParallelConfig {
        rank: 0,
        world_size: tp_size,
    };
    tensor_parallel.validate_for(&config)?;
    Qwen3ProjectionFusionPlan::resolve(
        options,
        &config,
        tensor_parallel,
        ProjectionFusionEnvironment {
            numeric_policy: pegainfer_kernels::ops::numeric_policy(),
            decode_overlap,
        },
    )
}

fn resolve_one(
    control: ProjectionFusionControl,
    kind: ProjectionKind,
    phase: ProjectionPhase,
    config: &Config,
    tensor_parallel: TensorParallelConfig,
    environment: ProjectionFusionEnvironment,
) -> Result<ProjectionFusionDecision> {
    match control {
        ProjectionFusionControl::Split => Ok(ProjectionFusionDecision::new(
            false,
            ProjectionFusionResolutionReason::ExplicitSplit,
        )),
        ProjectionFusionControl::Auto => {
            if let Some(reason) = unsupported_reason(config, tensor_parallel, environment) {
                return Ok(ProjectionFusionDecision::new(false, reason));
            }
            let fused = auto_whitelisted(kind, phase, config, tensor_parallel, environment);
            Ok(ProjectionFusionDecision::new(
                fused,
                if fused {
                    ProjectionFusionResolutionReason::AutoWhitelisted
                } else {
                    ProjectionFusionResolutionReason::AutoNotWhitelisted
                },
            ))
        }
        ProjectionFusionControl::ForceFused => {
            validate_force_supported(kind, phase, config, tensor_parallel, environment)?;
            Ok(ProjectionFusionDecision::new(
                true,
                ProjectionFusionResolutionReason::ForcedFused,
            ))
        }
    }
}

fn unsupported_reason(
    config: &Config,
    tensor_parallel: TensorParallelConfig,
    environment: ProjectionFusionEnvironment,
) -> Option<ProjectionFusionResolutionReason> {
    if !is_qwen3_4b_geometry(config) {
        return Some(ProjectionFusionResolutionReason::UnsupportedGeometry);
    }
    if !matches!(tensor_parallel.world_size, 1 | 2) {
        return Some(ProjectionFusionResolutionReason::UnsupportedTensorParallel);
    }
    if environment.numeric_policy != NumericPolicy::Tuned {
        return Some(ProjectionFusionResolutionReason::UnsupportedNumericPolicy);
    }
    if !matches!(environment.decode_overlap, DecodeOverlap::Off) {
        return Some(ProjectionFusionResolutionReason::UnsupportedDecodeOverlap);
    }
    let q_dim = config.local_q_dim(tensor_parallel);
    let kv_dim = config.local_kv_dim(tensor_parallel);
    let intermediate = config.local_intermediate_size(tensor_parallel);
    if q_dim == 0 || kv_dim == 0 || intermediate == 0 {
        return Some(ProjectionFusionResolutionReason::EmptyLocalProjection);
    }
    None
}

/// Production entries are populated only after the correctness and performance
/// matrix in `docs/models/qwen3/fused-projection-parity-plan.md` passes.
///
/// Keep this empty while the candidate paths are under construction.
const fn auto_whitelisted(
    _kind: ProjectionKind,
    _phase: ProjectionPhase,
    _config: &Config,
    _tensor_parallel: TensorParallelConfig,
    _environment: ProjectionFusionEnvironment,
) -> bool {
    false
}

fn validate_force_supported(
    kind: ProjectionKind,
    phase: ProjectionPhase,
    config: &Config,
    tensor_parallel: TensorParallelConfig,
    environment: ProjectionFusionEnvironment,
) -> Result<()> {
    anyhow::ensure!(
        is_qwen3_4b_geometry(config),
        "forced Qwen3 {} fusion is qualified only for Qwen3-4B geometry; \
         got hidden={}, q_heads={}, kv_heads={}, head_dim={}, intermediate={}",
        kind.label(),
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.intermediate_size,
    );
    anyhow::ensure!(
        matches!(tensor_parallel.world_size, 1 | 2),
        "forced Qwen3 {} fusion is qualified only for TP1/TP2; got phase={phase:?}, \
         rank={}, world_size={}",
        kind.label(),
        tensor_parallel.rank,
        tensor_parallel.world_size,
    );
    anyhow::ensure!(
        environment.numeric_policy == NumericPolicy::Tuned,
        "forced Qwen3 {} fusion requires NumericPolicy::Tuned; got phase={phase:?}, policy={:?}",
        kind.label(),
        environment.numeric_policy,
    );
    anyhow::ensure!(
        matches!(environment.decode_overlap, DecodeOverlap::Off),
        "forced Qwen3 {} fusion requires decode-overlap=off; got phase={phase:?}, overlap={:?}",
        kind.label(),
        environment.decode_overlap,
    );

    let q_dim = config.local_q_dim(tensor_parallel);
    let kv_dim = config.local_kv_dim(tensor_parallel);
    let intermediate = config.local_intermediate_size(tensor_parallel);
    anyhow::ensure!(
        q_dim > 0 && kv_dim > 0 && intermediate > 0,
        "forced Qwen3 {} fusion resolved an empty local projection: phase={phase:?}, \
         tp={}, q_dim={}, kv_dim={}, intermediate={}",
        kind.label(),
        tensor_parallel.world_size,
        q_dim,
        kv_dim,
        intermediate,
    );
    Ok(())
}

const fn is_qwen3_4b_geometry(config: &Config) -> bool {
    config.hidden_size == 2560
        && config.intermediate_size == 9728
        && config.num_attention_heads == 32
        && config.num_key_value_heads == 8
        && config.head_dim == 128
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3_4b() -> Config {
        Config {
            hidden_size: 2560,
            intermediate_size: 9728,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151_936,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1.0e6,
            max_position_embeddings: 40_960,
            eos_token_id: 151_645,
            tie_word_embeddings: true,
            stop_token_ids: vec![151_645],
        }
    }

    fn environment() -> ProjectionFusionEnvironment {
        ProjectionFusionEnvironment {
            numeric_policy: NumericPolicy::Tuned,
            decode_overlap: DecodeOverlap::Off,
        }
    }

    #[test]
    fn auto_stays_split_until_a_measured_whitelist_entry_exists() {
        for world_size in [1, 2] {
            let plan = Qwen3ProjectionFusionPlan::resolve(
                Qwen3ProjectionFusionOptions::default(),
                &qwen3_4b(),
                TensorParallelConfig {
                    rank: 0,
                    world_size,
                },
                environment(),
            )
            .unwrap();
            assert!(!plan.qkv(ProjectionPhase::Decode));
            assert!(!plan.gate_up(ProjectionPhase::Decode));
            assert!(!plan.qkv(ProjectionPhase::PrefillUnified));
            assert!(!plan.gate_up(ProjectionPhase::PrefillUnified));
            for decision in [
                plan.decode_qkv,
                plan.decode_gate_up,
                plan.prefill_unified_qkv,
                plan.prefill_unified_gate_up,
            ] {
                assert_eq!(
                    decision.reason,
                    ProjectionFusionResolutionReason::AutoNotWhitelisted
                );
            }
        }
    }

    #[test]
    fn force_supports_qwen3_4b_tp1_and_tp2() {
        for world_size in [1, 2] {
            let plan = Qwen3ProjectionFusionPlan::resolve(
                Qwen3ProjectionFusionOptions::force_fused(),
                &qwen3_4b(),
                TensorParallelConfig {
                    rank: 0,
                    world_size,
                },
                environment(),
            )
            .unwrap();
            assert!(plan.qkv(ProjectionPhase::Decode));
            assert!(plan.gate_up(ProjectionPhase::Decode));
            assert!(plan.qkv(ProjectionPhase::PrefillUnified));
            assert!(plan.gate_up(ProjectionPhase::PrefillUnified));
            for decision in [
                plan.decode_qkv,
                plan.decode_gate_up,
                plan.prefill_unified_qkv,
                plan.prefill_unified_gate_up,
            ] {
                assert_eq!(
                    decision.reason,
                    ProjectionFusionResolutionReason::ForcedFused
                );
            }
        }
    }

    #[test]
    fn qkv_and_gate_up_controls_resolve_independently() {
        let plan = Qwen3ProjectionFusionPlan::resolve(
            Qwen3ProjectionFusionOptions {
                qkv: ProjectionFusionControl::ForceFused,
                gate_up: ProjectionFusionControl::Split,
            },
            &qwen3_4b(),
            TensorParallelConfig::default(),
            environment(),
        )
        .unwrap();
        for phase in [ProjectionPhase::Decode, ProjectionPhase::PrefillUnified] {
            assert!(plan.qkv(phase));
            assert!(!plan.gate_up(phase));
        }
        assert_eq!(
            plan.decode_qkv.reason,
            ProjectionFusionResolutionReason::ForcedFused
        );
        assert_eq!(
            plan.decode_gate_up.reason,
            ProjectionFusionResolutionReason::ExplicitSplit
        );
        assert_eq!(
            serde_json::to_value(plan).unwrap(),
            serde_json::json!({
                "decode_qkv": {"fused": true, "reason": "forced_fused"},
                "decode_gate_up": {"fused": false, "reason": "explicit_split"},
                "prefill_unified_qkv": {"fused": true, "reason": "forced_fused"},
                "prefill_unified_gate_up": {"fused": false, "reason": "explicit_split"},
            })
        );
    }

    #[test]
    fn auto_reports_the_first_unsupported_qualification_reason() {
        let mut other_geometry = qwen3_4b();
        other_geometry.hidden_size = 4096;
        let cases = [
            (
                other_geometry,
                TensorParallelConfig::default(),
                environment(),
                ProjectionFusionResolutionReason::UnsupportedGeometry,
            ),
            (
                qwen3_4b(),
                TensorParallelConfig {
                    rank: 0,
                    world_size: 4,
                },
                environment(),
                ProjectionFusionResolutionReason::UnsupportedTensorParallel,
            ),
            (
                qwen3_4b(),
                TensorParallelConfig::default(),
                ProjectionFusionEnvironment {
                    numeric_policy: NumericPolicy::Pin,
                    decode_overlap: DecodeOverlap::Off,
                },
                ProjectionFusionResolutionReason::UnsupportedNumericPolicy,
            ),
            (
                qwen3_4b(),
                TensorParallelConfig::default(),
                ProjectionFusionEnvironment {
                    numeric_policy: NumericPolicy::Tuned,
                    decode_overlap: DecodeOverlap::SharedSm,
                },
                ProjectionFusionResolutionReason::UnsupportedDecodeOverlap,
            ),
        ];
        for (config, tensor_parallel, environment, expected) in cases {
            let plan = Qwen3ProjectionFusionPlan::resolve(
                Qwen3ProjectionFusionOptions::default(),
                &config,
                tensor_parallel,
                environment,
            )
            .unwrap();
            for decision in [
                plan.decode_qkv,
                plan.decode_gate_up,
                plan.prefill_unified_qkv,
                plan.prefill_unified_gate_up,
            ] {
                assert!(!decision.fused);
                assert_eq!(decision.reason, expected);
            }
        }
    }

    #[test]
    fn force_fails_closed_for_tp_greater_than_two() {
        let error = Qwen3ProjectionFusionPlan::resolve(
            Qwen3ProjectionFusionOptions::force_fused(),
            &qwen3_4b(),
            TensorParallelConfig {
                rank: 0,
                world_size: 4,
            },
            environment(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("only for TP1/TP2"), "{error}");
    }

    #[test]
    fn force_fails_closed_for_other_geometry() {
        let mut config = qwen3_4b();
        config.hidden_size = 4096;
        let error = Qwen3ProjectionFusionPlan::resolve(
            Qwen3ProjectionFusionOptions::force_fused(),
            &config,
            TensorParallelConfig::default(),
            environment(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("only for Qwen3-4B geometry"), "{error}");
    }

    #[test]
    fn force_fails_closed_for_non_tuned_policy_and_overlap() {
        for numeric_policy in [NumericPolicy::Pin, NumericPolicy::PerToken] {
            let error = Qwen3ProjectionFusionPlan::resolve(
                Qwen3ProjectionFusionOptions::force_fused(),
                &qwen3_4b(),
                TensorParallelConfig::default(),
                ProjectionFusionEnvironment {
                    numeric_policy,
                    decode_overlap: DecodeOverlap::Off,
                },
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("NumericPolicy::Tuned"), "{error}");
        }

        for decode_overlap in [
            DecodeOverlap::SharedSm,
            DecodeOverlap::GreenCtx { decode_pct: 20 },
        ] {
            let error = Qwen3ProjectionFusionPlan::resolve(
                Qwen3ProjectionFusionOptions::force_fused(),
                &qwen3_4b(),
                TensorParallelConfig::default(),
                ProjectionFusionEnvironment {
                    numeric_policy: NumericPolicy::Tuned,
                    decode_overlap,
                },
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("decode-overlap=off"), "{error}");
        }
    }
}
