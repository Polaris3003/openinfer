#!/usr/bin/env bash
# Current-head GPU gate for issue #746 reviewer evidence.
#
# Usage:
#   bash scripts/run_qwen3_746_projection_ab.sh
#   bash scripts/run_qwen3_746_projection_ab.sh --shutdown-after
#
# Optional overrides:
#   REPO_ROOT=/root/openinfer
#   MODEL_4B=/root/autodl-tmp/models/Qwen3-4B
#   MODEL_8B=/root/autodl-tmp/models/Qwen3-8B
#   RESULT_BASE=/root/autodl-tmp/pegainfer-results
#   BLOCKS=10

set -Eeuo pipefail

SHUTDOWN_AFTER=false
if [[ ${1:-} == --shutdown-after ]]; then
    SHUTDOWN_AFTER=true
    shift
fi
if (( $# != 0 )); then
    echo "usage: bash $0 [--shutdown-after]" >&2
    exit 2
fi

REPO_ROOT=${REPO_ROOT:-/root/openinfer}
MODEL_4B=${MODEL_4B:-/root/autodl-tmp/models/Qwen3-4B}
MODEL_8B=${MODEL_8B:-/root/autodl-tmp/models/Qwen3-8B}
RESULT_BASE=${RESULT_BASE:-/root/autodl-tmp/pegainfer-results}
BLOCKS=${BLOCKS:-10}
RUN_ID=$(date +%Y%m%d-%H%M%S)-$$
RUN_ROOT=${RESULT_BASE}/qwen3-746-projection-ab-${RUN_ID}
ARCHIVE=${RUN_ROOT}.tar.gz
FINAL_STATUS=FAILED

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/reports" "${RUN_ROOT}/meta"
exec > >(tee -a "${RUN_ROOT}/runner.console.log") 2>&1

archive_on_exit() {
    local rc=$?
    trap - EXIT ERR INT TERM
    set +e
    printf '%s\n' "${FINAL_STATUS}" > "${RUN_ROOT}/runner-status.txt"
    tar -C "$(dirname "${RUN_ROOT}")" -czf "${ARCHIVE}" "$(basename "${RUN_ROOT}")"
    local tar_rc=$?
    if (( tar_rc == 0 )); then
        sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
    else
        rc=${tar_rc}
        FINAL_STATUS=FAILED
    fi
    echo
    echo "FINAL_STATUS=${FINAL_STATUS}"
    echo "RUN_ROOT=${RUN_ROOT}"
    echo "ARCHIVE=${ARCHIVE}"
    echo "SHA256=${ARCHIVE}.sha256"
    if [[ ${SHUTDOWN_AFTER} == true ]]; then
        sync
        /usr/bin/shutdown -h now
    fi
    exit "${rc}"
}

report_error() {
    local rc=$?
    printf 'FAILED_LINE=%s\nFAILED_RC=%s\nFAILED_COMMAND=%s\n' \
        "$1" "${rc}" "$2" | tee "${RUN_ROOT}/failure.txt" >&2
    return "${rc}"
}

trap archive_on_exit EXIT
trap 'report_error "${LINENO}" "${BASH_COMMAND}"' ERR
trap 'exit 130' INT TERM

run_log() {
    local name=$1
    shift
    echo
    echo "===== ${name} ====="
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
    set +e
    "$@" 2>&1 | tee "${RUN_ROOT}/logs/${name}.log"
    local rc=${PIPESTATUS[0]}
    set -e
    return "${rc}"
}

for command in git cargo rustc nvcc nvidia-smi python3 tar sha256sum; do
    command -v "${command}" >/dev/null || {
        echo "missing required command: ${command}" >&2
        exit 2
    }
done
[[ $(uname -s) == Linux ]] || { echo "Linux is required" >&2; exit 2; }
[[ -f ${REPO_ROOT}/Cargo.toml ]] || { echo "repo missing: ${REPO_ROOT}" >&2; exit 2; }
[[ -f ${MODEL_4B}/config.json ]] || { echo "4B model missing: ${MODEL_4B}" >&2; exit 2; }
[[ -f ${MODEL_8B}/config.json ]] || { echo "8B model missing: ${MODEL_8B}" >&2; exit 2; }
[[ ${BLOCKS} =~ ^[0-9]+$ ]] && (( BLOCKS >= 10 )) || {
    echo "BLOCKS must be an integer >= 10" >&2
    exit 2
}

cd "${REPO_ROOT}"
git status --short > "${RUN_ROOT}/meta/git-status.txt"
{
    git diff --name-only
    git diff --cached --name-only
} | sort -u > "${RUN_ROOT}/meta/tracked-dirty-paths.txt"
UNEXPECTED_DIRTY=$(
    grep -v '^test_data/qwen3-4b-lora-golden\.safetensors$' \
        "${RUN_ROOT}/meta/tracked-dirty-paths.txt" || true
)
if [[ -n ${UNEXPECTED_DIRTY} ]]; then
    echo "unexpected tracked changes; refusing to benchmark modified code:" >&2
    printf '%s\n' "${UNEXPECTED_DIRTY}" >&2
    exit 2
fi
if [[ -f test_data/qwen3-4b-lora-golden.safetensors ]]; then
    sha256sum test_data/qwen3-4b-lora-golden.safetensors \
        > "${RUN_ROOT}/meta/lora-fixture.sha256"
fi

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
(( GPU_COUNT >= 2 )) || { echo "two GPUs are required for TP2 gates" >&2; exit 2; }
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits -i 0 \
    | head -n 1 | tr -d ' .')
[[ ${COMPUTE_CAP} =~ ^[0-9]+$ ]] || { echo "invalid compute capability" >&2; exit 2; }

export PEGAINFER_CUDA_SM=${COMPUTE_CAP}
export RUST_LOG=info
export CARGO_INCREMENTAL=0

{
    echo "branch=$(git branch --show-current)"
    echo "commit=$(git rev-parse HEAD)"
    echo "rustc=$(rustc --version)"
    echo "cargo=$(cargo --version)"
    echo "nvcc=$(nvcc --version | tail -n 1)"
    echo "PEGAINFER_CUDA_SM=${PEGAINFER_CUDA_SM}"
    nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total \
        --format=csv,noheader
} | tee "${RUN_ROOT}/meta/environment.txt"

run_log fmt cargo fmt --all -- --check
run_log metadata cargo metadata --locked --no-deps --format-version 1
run_log clippy env CUDA_VISIBLE_DEVICES=0 cargo clippy --release --locked \
    -p pegainfer-qwen3 --features kernel-report --lib --bin qwen3_kernel_report -- -D warnings
run_log selector-tests env CUDA_VISIBLE_DEVICES=0 cargo test --release --locked \
    -p pegainfer-qwen3 --lib projection_fusion::tests -- --nocapture
run_log split-qkv-tests env CUDA_VISIBLE_DEVICES=0 cargo test --release --locked \
    -p pegainfer-kernels --lib split_qkv_ -- --nocapture

run_log projection-ab env CUDA_VISIBLE_DEVICES=0 cargo run --release --locked \
    -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report -- \
    projection-ab --models qwen3-4b,qwen3-8b --batch-sizes 1,8 \
    --blocks "${BLOCKS}" --out "${RUN_ROOT}/reports/projection-ab.json"

run_log hf-golden-4b env CUDA_VISIBLE_DEVICES=0,1 \
    PEGAINFER_TEST_MODEL_PATH="${MODEL_4B}" cargo test --release --locked \
    -p pegainfer-qwen3 --test hf_golden_gate -- --nocapture
run_log hf-golden-8b env CUDA_VISIBLE_DEVICES=0,1 \
    PEGAINFER_TEST_MODEL_PATH="${MODEL_8B}" cargo test --release --locked \
    -p pegainfer-qwen3 --test hf_golden_gate -- --nocapture
run_log lora-golden-4b env CUDA_VISIBLE_DEVICES=0,1 \
    PEGAINFER_TEST_MODEL_PATH="${MODEL_4B}" cargo test --release --locked \
    -p pegainfer-qwen3 --test lora_golden_gate -- --nocapture

python3 - "${RUN_ROOT}/reports/projection-ab.json" <<'PY' \
    | tee "${RUN_ROOT}/reports/projection-ab-summary.txt"
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
for case in report["cases"]:
    print(f'{case["model"]} batch={case["batch_size"]}')
    for comparison in case["comparisons"]:
        summary = comparison["summary"]
        ci = summary["paired_improvement_pct_bootstrap_ci95"]
        print(
            f'  {comparison["name"]}: '
            f'mean={summary["paired_improvement_pct_mean"]:+.3f}% '
            f'median={summary["paired_improvement_pct_median"]:+.3f}% '
            f'CI95=[{ci[0]:+.3f}%, {ci[1]:+.3f}%]'
        )
    copy = case["split_qkv"]["summary"]
    print(f'  split_qkv standalone: mean={copy["mean_us"]:.3f}us p95={copy["p95_us"]:.3f}us')
PY

sha256sum "${RUN_ROOT}/reports/projection-ab.json" \
    > "${RUN_ROOT}/reports/projection-ab.json.sha256"
FINAL_STATUS=PASSED
