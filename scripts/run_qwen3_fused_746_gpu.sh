#!/usr/bin/env bash
# End-to-end GPU evidence runner for pegainfer issue #746.
#
# Usage:
#   bash scripts/run_qwen3_fused_746_gpu.sh          # smoke + full 104-command suite
#   bash scripts/run_qwen3_fused_746_gpu.sh --smoke-only
#
# The runner is intentionally non-destructive: it never cleans the worktree,
# overwrites the tracked LoRA fixture, or reuses an old artifact directory.

set -Eeuo pipefail

REPO_ROOT=/root/openinfer
MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B
RESULT_BASE=/root/pegainfer-results
FIXTURE_PATH=${REPO_ROOT}/test_data/qwen3-4b-lora-golden.safetensors
MINIMUM_COMMIT=df0027dd3d27527b0b1f650d7b263412df3015f8
EXPECTED_BRANCH=feat/qwen3-fused-projection-parity-v2
MODE=all

if [[ ${1:-} == "--smoke-only" ]]; then
    MODE=smoke
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--smoke-only]" >&2
    exit 2
fi

for command in git python3 nvidia-smi nvcc rustc cargo timeout sha256sum tar; do
    command -v "${command}" >/dev/null || {
        echo "missing required command: ${command}" >&2
        exit 2
    }
done

[[ $(uname -s) == Linux ]] || {
    echo "this runner requires Linux" >&2
    exit 2
}
[[ -f ${REPO_ROOT}/Cargo.toml ]] || {
    echo "repository not found: ${REPO_ROOT}" >&2
    exit 2
}
[[ -f ${MODEL_PATH}/config.json ]] || {
    echo "model config not found: ${MODEL_PATH}/config.json" >&2
    exit 2
}
[[ -f ${FIXTURE_PATH} ]] || {
    echo "LoRA fixture not found: ${FIXTURE_PATH}" >&2
    exit 2
}

cd "${REPO_ROOT}"

branch=$(git branch --show-current)
commit=$(git rev-parse HEAD)
[[ ${branch} == "${EXPECTED_BRANCH}" ]] || {
    echo "wrong branch: ${branch}; expected ${EXPECTED_BRANCH}" >&2
    exit 2
}
git merge-base --is-ancestor "${MINIMUM_COMMIT}" HEAD || {
    echo "HEAD ${commit} does not contain required ${MINIMUM_COMMIT}" >&2
    exit 2
}

tracked_dirty=$(git status --short --untracked-files=no)
if [[ ${tracked_dirty} != " M test_data/qwen3-4b-lora-golden.safetensors" ]]; then
    echo "unexpected tracked changes; refusing to mix them into evidence:" >&2
    printf '%s\n' "${tracked_dirty}" >&2
    exit 2
fi

mapfile -t gpu_rows < <(
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader,nounits
)
[[ ${#gpu_rows[@]} -ge 2 ]] || {
    echo "two visible GPUs are required; found ${#gpu_rows[@]}" >&2
    exit 2
}
gpu0=$(cut -d, -f2- <<<"${gpu_rows[0]}" | xargs)
gpu1=$(cut -d, -f2- <<<"${gpu_rows[1]}" | xargs)
[[ ${gpu0} == "${gpu1}" ]] || {
    echo "GPU 0/1 differ: ${gpu_rows[0]} vs ${gpu_rows[1]}" >&2
    exit 2
}
compute_cap=$(cut -d, -f3 <<<"${gpu_rows[0]}" | xargs)

export CUDA_VISIBLE_DEVICES=0,1
export PEGAINFER_CUDA_SM=${compute_cap//./}
export RUST_LOG=info

mkdir -p "${RESULT_BASE}"
RUN_ROOT=$(mktemp -d "${RESULT_BASE}/qwen3-fused-746-XXXXXX")
SUITE_DIR=${RUN_ROOT}/suite
SMOKE_JSON=${RUN_ROOT}/smoke-tp1-decode-split.json
ARCHIVE=${RUN_ROOT}.tar.gz
export REPO_ROOT MODEL_PATH RESULT_BASE RUN_ROOT SUITE_DIR

archive_results() {
    local status=${1:-unknown}
    printf '%s\n' "${status}" > "${RUN_ROOT}/runner-status.txt"
    git status --short > "${RUN_ROOT}/git-status-after.txt" || true
    tar -C "$(dirname "${RUN_ROOT}")" \
        -czf "${ARCHIVE}" "$(basename "${RUN_ROOT}")" || true
    if [[ -f ${ARCHIVE} ]]; then
        sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256" || true
    fi
    echo
    echo "RUN_ROOT=${RUN_ROOT}"
    echo "ARCHIVE=${ARCHIVE}"
    echo "STATUS=${status}"
}

on_error() {
    local rc=$?
    local line=${BASH_LINENO[0]:-unknown}
    trap - ERR INT TERM
    echo "runner failed: rc=${rc} line=${line}" >&2
    archive_results "FAILED rc=${rc} line=${line}"
    exit "${rc}"
}
trap on_error ERR INT TERM

printf '%s\n' \
    "export REPO_ROOT=${REPO_ROOT}" \
    "export MODEL_PATH=${MODEL_PATH}" \
    "export RESULT_BASE=${RESULT_BASE}" \
    "export RUN_ROOT=${RUN_ROOT}" \
    "export SUITE_DIR=${SUITE_DIR}" \
    "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
    "export PEGAINFER_CUDA_SM=${PEGAINFER_CUDA_SM}" \
    "export RUST_LOG=${RUST_LOG}" \
    > "${RUN_ROOT}/run.env"

exec > >(tee -a "${RUN_ROOT}/runner.console.log") 2>&1

echo "===== issue #746 GPU runner ====="
echo "mode=${MODE}"
echo "repo=${REPO_ROOT}"
echo "model=${MODEL_PATH}"
echo "branch=${branch}"
echo "commit=${commit}"
echo "gpu=${gpu0}"
echo "PEGAINFER_CUDA_SM=${PEGAINFER_CUDA_SM}"
echo "RUN_ROOT=${RUN_ROOT}"

{
    date -Is
    uname -a
    git remote -v
    git status --short
    rustc --version --verbose
    cargo --version
    nvcc --version
    python3 --version
    nvidia-smi
} > "${RUN_ROOT}/environment.txt" 2>&1

echo "===== validating model and fixture ====="
python3 - "${MODEL_PATH}/config.json" "${FIXTURE_PATH}" <<'PY'
import json
import re
import sys
from collections import Counter
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
assert config.get("model_type") == "qwen3", config.get("model_type")
assert int(config["num_hidden_layers"]) == 36, config["num_hidden_layers"]

fixture = Path(sys.argv[2])
with fixture.open("rb") as handle:
    header_len = int.from_bytes(handle.read(8), "little")
    header = json.loads(handle.read(header_len))
metadata = header.get("__metadata__", {})
targets = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
assert set(json.loads(metadata["target_modules"])) == targets
assert metadata.get("model") == "Qwen3-4B", metadata.get("model")
effect = float(metadata["mean_effect_nat"])
assert 0.1 <= effect <= 2.0, effect

pattern = re.compile(
    r"^adapter/base_model\.model\.model\.layers\.(\d+)\."
    r"(self_attn|mlp)\.(q_proj|k_proj|v_proj|gate_proj|up_proj)\."
    r"lora_([AB])\.weight$"
)
coverage = Counter()
for name in header:
    if not name.startswith("adapter/"):
        continue
    match = pattern.fullmatch(name)
    assert match is not None, name
    layer, _block, target, side = match.groups()
    coverage[(int(layer), target, side)] += 1
expected = {
    (layer, target, side)
    for layer in range(36)
    for target in targets
    for side in ("A", "B")
}
assert set(coverage) == expected
assert all(count == 1 for count in coverage.values())
print(f"fixture=PASS effect={effect} adapter_tensors={len(coverage)}")
PY

python3 tools/validation/qwen3_fused_projection_suite.py check-fixture \
    --path "${FIXTURE_PATH}"
cp "${FIXTURE_PATH}" "${RUN_ROOT}/qwen3-4b-lora-golden.safetensors"
sha256sum "${RUN_ROOT}/qwen3-4b-lora-golden.safetensors" \
    | tee "${RUN_ROOT}/fixture.sha256"

echo "===== CPU-side preflight ====="
cargo fmt --all --check
python3 -m unittest tools/validation/test_qwen3_fused_projection_suite.py
python3 -m py_compile \
    tools/validation/qwen3_fused_projection_suite.py \
    scripts/bench_http_serving.py

echo "===== visible release prebuild ====="
echo "This can take several minutes on the first run; compiler output is live."
timeout --signal=INT --kill-after=30s 30m \
    cargo build --release -p pegainfer-server 2>&1 \
    | tee "${RUN_ROOT}/server-build.log"

echo "===== checking benchmark port ====="
python3 - <<'PY'
import socket

sock = socket.socket()
try:
    sock.bind(("127.0.0.1", 18080))
finally:
    sock.close()
print("port 18080 available")
PY

echo "===== real HTTP smoke ====="
echo "The cell now prints a heartbeat every 15 seconds while the server starts."
timeout --signal=INT --kill-after=30s 20m \
    python3 tools/validation/qwen3_fused_projection_suite.py benchmark-cell \
        --model-path "${MODEL_PATH}" \
        --tp-size 1 \
        --qkv-fusion split \
        --gate-up-fusion split \
        --out "${SMOKE_JSON}" \
        --port 18080 \
        --prompt-words 1024 \
        --output-len 32 \
        --concurrency 1 \
        --warmup 1 \
        --iters 2 \
        --seed 42 \
        --ready-timeout 600 \
        --request-timeout 600

python3 - "${SMOKE_JSON}" <<'PY' | tee "${RUN_ROOT}/smoke-check.txt"
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
summary = report["summary"]
trace = report["server_trace"]
tpot = report["metrics"]["tpot"]
assert summary["completed"] == 2, summary
assert summary["failed"] == 0, summary
assert summary["timeouts"] == 0, summary
assert summary["output_token_count_source"] == "server_trace.completion_tokens", summary
assert summary["output_tokens_total"] == 64, summary
assert trace["coverage_ratio"] == 1.0, trace
assert trace["token_timing_coverage_ratio"] == 1.0, trace
assert trace["prompt_tokens"]["samples"] == 2, trace
assert trace["completion_tokens"]["samples"] == 2, trace
assert tpot["samples"] == 2 and tpot["p50_ms"] is not None, tpot
print(json.dumps({
    "completed": summary["completed"],
    "output_tokens_total": summary["output_tokens_total"],
    "trace_coverage": trace["coverage_ratio"],
    "token_timing_coverage": trace["token_timing_coverage_ratio"],
    "tpot": tpot,
}, indent=2))
PY

if [[ ${MODE} == smoke ]]; then
    trap - ERR INT TERM
    archive_results "SMOKE_PASS"
    exit 0
fi

echo "===== schema-v3 dry run ====="
DRY_RUN_DIR=${RUN_ROOT}/dry-run
python3 tools/validation/qwen3_fused_projection_suite.py run \
    --model-path "${MODEL_PATH}" \
    --output-dir "${DRY_RUN_DIR}" \
    --tp-sizes 1,2 \
    --concurrency 1,8 \
    --benchmark-port 18080 \
    --dry-run > "${RUN_ROOT}/dry-run.console.log"
python3 - "${DRY_RUN_DIR}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest["schema"] == 3
assert manifest["status"] == "planned"
assert len(manifest["commands"]) == 104
print("dry_run=PASS schema=3 commands=104")
PY

echo "===== full 104-command suite ====="
echo "This is a long run. Each command prints its index; server startup prints heartbeats."
set +e
python3 tools/validation/qwen3_fused_projection_suite.py run \
    --model-path "${MODEL_PATH}" \
    --output-dir "${SUITE_DIR}" \
    --sections correctness,projection,topology,benchmark \
    --tp-sizes 1,2 \
    --concurrency 1,8 \
    --warmup 5 \
    --iters 20 \
    --seed 42 \
    --benchmark-port 18080 \
    --ready-timeout 600 \
    --request-timeout 600 \
    --projection-warmup 2 \
    --projection-iters 5 \
    --topology-batches 1,8,32,64 \
    --topology-kv-len 2048 \
    --topology-iters 32 \
    --model-hash full \
    --no-fail-fast
SUITE_RC=$?
set -e
printf 'suite_rc=%s\n' "${SUITE_RC}" | tee "${RUN_ROOT}/suite-exit.txt"

echo "===== final summarize ====="
set +e
python3 tools/validation/qwen3_fused_projection_suite.py summarize \
    --output-dir "${SUITE_DIR}" 2>&1 | tee "${RUN_ROOT}/summarize.console.log"
SUMMARIZE_RC=${PIPESTATUS[0]}
set -e
printf 'summarize_rc=%s\n' "${SUMMARIZE_RC}" | tee "${RUN_ROOT}/summarize-exit.txt"

python3 - "${SUITE_DIR}" <<'PY' | tee "${RUN_ROOT}/final-review.txt"
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
summary = json.loads((root / "summary.json").read_text())
commands = manifest["commands"]
print("schema=", summary["schema"])
print("overall_status=", summary["overall_status"])
print("manifest_status=", manifest["status"])
print("command_count=", len(commands))
print("returncodes=", dict(Counter(entry.get("returncode") for entry in commands)))
print("correctness=", summary["correctness"]["passed"])
print("projection=", summary["projection"]["passed"])
print("topology=", summary["topology"]["passed"])
print("benchmark_observed=", summary["benchmark"]["observed_cells_with_repeats"])
print("benchmark_required=", summary["benchmark"]["required_cells_with_repeats"])
print("benchmark_missing=", len(summary["benchmark"]["missing"]))
print("benchmark_errors=", len(summary["benchmark"]["errors"]))
for row in summary["decisions"]:
    print(
        f"{row['projection']} {row['phase']} TP{row['tp_size']} "
        f"decision={row['decision']} mean={row['mean_improvement_pct']} "
        f"direction={row['direction_consistent']} "
        f"throughput={row['throughput_consistent']} "
        f"kernel={row['kernel_direction_pass']} "
        f"correctness={row['correctness_pass']}"
    )
assert summary["schema"] == 3
assert len(commands) == 104
PY

trap - ERR INT TERM
if [[ ${SUITE_RC} -eq 0 && ${SUMMARIZE_RC} -eq 0 ]]; then
    archive_results "FULL_PASS"
    exit 0
fi
archive_results "FULL_INCOMPLETE suite_rc=${SUITE_RC} summarize_rc=${SUMMARIZE_RC}"
exit 1
