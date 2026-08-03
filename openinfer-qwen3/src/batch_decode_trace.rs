#[cfg(feature = "kernel-call-trace")]
use anyhow::Result;
#[cfg(feature = "kernel-call-trace")]
use openinfer_core::ops::call_trace;
#[cfg(feature = "kernel-call-trace")]
use openinfer_kernels::ops::NumericPolicy;
#[cfg(feature = "kernel-call-trace")]
use openinfer_kernels::ops::numeric_policy;
#[cfg(feature = "kernel-call-trace")]
use openinfer_kernels::tensor::KernelCall;
#[cfg(feature = "kernel-call-trace")]
use openinfer_kv_cache::KvView;

#[cfg(feature = "kernel-call-trace")]
use crate::batch_decode_buffers::BatchDecodeBuffers;
#[cfg(feature = "kernel-call-trace")]
use crate::weights::ModelRuntimeConfig;
#[cfg(feature = "kernel-call-trace")]
use crate::weights::Qwen3Model;

pub const MODEL: &str = "qwen3";
pub const PHASE_DECODE: &str = "decode";
pub const NUM_LAYERS: usize = 36;
pub const NUM_Q_HEADS: usize = 32;
pub const NUM_KV_HEADS: usize = 8;
pub const HEAD_DIM_VALUE: usize = 128;
pub const KV_DIM_VALUE: usize = NUM_KV_HEADS * HEAD_DIM_VALUE;
pub const RMS_NORM_EPS: f32 = 1.0e-6;

fn trace_kv_num_blocks(kv_len: usize, block_size: usize) -> usize {
    // Block 0 is the CUDA Graph padding page. Topology tracing needs only one
    // physical page table because it does not validate request-local KV data.
    1 + kv_len.div_ceil(block_size)
}

#[cfg(feature = "kernel-call-trace")]
pub fn trace_decode_kernel_calls(
    model_path: &str,
    batch_size: usize,
    kv_len: usize,
) -> Result<Vec<KernelCall>> {
    trace_decode_kernel_calls_with_projection_fusion(
        model_path,
        batch_size,
        kv_len,
        crate::Qwen3ProjectionFusionOptions::default(),
    )
}

#[cfg(feature = "kernel-call-trace")]
pub fn trace_decode_kernel_calls_with_projection_fusion(
    model_path: &str,
    batch_size: usize,
    kv_len: usize,
    projection_fusion: crate::Qwen3ProjectionFusionOptions,
) -> Result<Vec<KernelCall>> {
    trace_decode_kernel_calls_inner(model_path, batch_size, kv_len, projection_fusion, false)
}

#[cfg(feature = "kernel-call-trace")]
pub fn trace_decode_topology_calls_with_projection_fusion(
    model_path: &str,
    batch_size: usize,
    kv_len: usize,
    projection_fusion: crate::Qwen3ProjectionFusionOptions,
) -> Result<Vec<KernelCall>> {
    trace_decode_kernel_calls_inner(model_path, batch_size, kv_len, projection_fusion, true)
}

#[cfg(feature = "kernel-call-trace")]
fn trace_decode_kernel_calls_inner(
    model_path: &str,
    batch_size: usize,
    kv_len: usize,
    projection_fusion: crate::Qwen3ProjectionFusionOptions,
    shared_kv_pages: bool,
) -> Result<Vec<KernelCall>> {
    anyhow::ensure!(batch_size > 0, "batch_size must be greater than zero");
    anyhow::ensure!(kv_len > 0, "kv_len must be greater than zero");

    let model = Qwen3Model::from_safetensors_with_runtime(
        model_path,
        ModelRuntimeConfig {
            enable_cuda_graph: false,
            tensor_parallel: None,
            device_ordinal: 0,
            projection_fusion,
            ..Default::default()
        },
    )?;
    let budget = model.kv_budget();
    let trace_num_blocks = if shared_kv_pages {
        trace_kv_num_blocks(kv_len, budget.block_size)
    } else {
        budget.num_blocks
    };
    let kv_mgr = openinfer_kv_cache::KvCacheManager::new(
        &model.device_ctx().stream,
        budget.num_layers,
        budget.num_kv_heads,
        budget.head_dim,
        budget.block_size,
        trace_num_blocks,
    )?;
    let layout = openinfer_core::kv_pool::KvLayout::new(
        budget.num_layers,
        budget.num_kv_heads,
        budget.head_dim,
        budget.block_size,
    );

    let request_kvs = if shared_kv_pages {
        None
    } else {
        // max_output_tokens must be at least 2: `apply_prefill` emits the first
        // generated token, and tracing a decode step needs one more.
        let dummy_prompt_len = if kv_len > 1 { kv_len - 1 } else { 1 };
        Some(
            (0..batch_size)
                .map(|_| {
                    let mut rkv = kv_mgr
                        .pool()
                        .new_request(vec![0; dummy_prompt_len], 2, None);
                    rkv.schedule_prefill(dummy_prompt_len, kv_mgr.pool())
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    rkv.apply_prefill(0, kv_mgr.pool())?;
                    rkv.schedule_decode(kv_mgr.pool())
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    Ok(rkv)
                })
                .collect::<Result<Vec<_>>>()?,
        )
    };
    let views = if let Some(request_kvs) = request_kvs.as_ref() {
        request_kvs.iter().map(|rkv| rkv.decode_view()).collect()
    } else {
        // Topology reports do not validate request-local KV contents. Reuse one
        // in-range page table for every synthetic row so bs=64/kv=2048 keeps
        // production tensor shapes without allocating 128 full KV blocks.
        let shared_pages = (1..trace_num_blocks)
            .map(|page| i32::try_from(page).expect("trace page index exceeds i32"))
            .collect();
        let shared_view = KvView::new(shared_pages, kv_len, budget.block_size);
        vec![shared_view; batch_size]
    };

    let mut bufs = BatchDecodeBuffers::new(
        model.device_ctx(),
        model.config().hidden_size,
        model.local_q_dim(),
        model.local_kv_dim(),
        model.local_intermediate_size(),
        model.config().vocab_size,
        batch_size,
        layout.page_size,
        kv_mgr.pool().padding_block_id(),
        model.local_num_attention_heads(),
        model.config().max_position_embeddings,
        model.fused_qkv(crate::projection_fusion::ProjectionPhase::Decode),
        model.fused_gate_up(crate::projection_fusion::ProjectionPhase::Decode),
    )?;
    // This trace path bypasses the serving executor (which warms the pinned shapes at startup);
    // warm here, outside `collect_result` below so it isn't recorded as a kernel call.
    if numeric_policy() == NumericPolicy::Pin {
        crate::batch_decode_buffers::warmup_decode_projection_pins(
            model.config().hidden_size,
            model.local_q_dim(),
            model.local_kv_dim(),
            model.local_intermediate_size(),
            model.config().vocab_size,
        )?;
    }
    let token_ids = vec![0_u32; batch_size];
    let ((), calls) = call_trace::collect_result(|| {
        model.batch_decode(
            &token_ids,
            &views,
            &vec![None; batch_size],
            kv_mgr.buffer().buffer(),
            &layout,
            &mut bufs,
            crate::batch_decode::DecodeGraphUse::Serve,
        )
    })?;
    Ok(calls)
}

pub fn normalize_call_site(label: &str) -> String {
    let Some(rest) = label.strip_prefix('L') else {
        return label.to_string();
    };
    let digit_count = rest
        .as_bytes()
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digit_count == 0 || rest.as_bytes().get(digit_count) != Some(&b'.') {
        return label.to_string();
    }
    format!("layer.*{}", &rest[digit_count..])
}

#[cfg(test)]
mod tests {
    use super::trace_kv_num_blocks;

    #[test]
    fn topology_trace_kv_capacity_is_independent_of_batch_size() {
        assert_eq!(trace_kv_num_blocks(2_048, 1_024), 3);
        assert_eq!(trace_kv_num_blocks(2_049, 1_024), 4);
    }
}
