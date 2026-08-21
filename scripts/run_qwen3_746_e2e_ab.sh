#!/usr/bin/env bash
# Current-head end-to-end A/B for Qwen3 issue #746.
#
# The baseline is the commit immediately before the original projection-fusion
# commits. The candidate is the current checkout. Both revisions are built in
# clean worktrees so untracked files cannot enter the evidence.
#
# Usage:
#   bash scripts/run_qwen3_746_e2e_ab.sh
#   bash scripts/run_qwen3_746_e2e_ab.sh --shutdown-after

set -Eeuo pipefail

REPO_ROOT=${REPO_ROOT:-/root/openinfer}
MODEL_4B=${MODEL_4B:-/root/autodl-tmp/models/Qwen3-4B}
MODEL_8B=${MODEL_8B:-/root/autodl-tmp/models/Qwen3-8B}
RESULT_BASE=${RESULT_BASE:-/root/autodl-tmp/pegainfer-results}
WORK_BASE=${WORK_BASE:-/root/autodl-tmp/pegainfer-validation-worktrees}
BASELINE_COMMIT=${BASELINE_COMMIT:-1c596322717dc7e3a8f0dbfb41774f12f7a36074}
GPU=${GPU:-0}
PORT=${PORT:-18086}
DIRECT_ITERS=${DIRECT_ITERS:-50}
HTTP_REQUESTS_PER_CONCURRENCY=${HTTP_REQUESTS_PER_CONCURRENCY:-5}
READY_TIMEOUT=${READY_TIMEOUT:-900}
KEEP_WORKTREES=0
SHUTDOWN_AFTER=0

usage() {
    sed -n '1,12p' "$0"
    cat <<'EOF'

Options:
  --model-4b PATH       Qwen3-4B model directory
  --model-8b PATH       Qwen3-8B model directory
  --gpu ORDINAL         GPU ordinal for TP1 (default: 0)
  --result-base PATH    Result directory on the data disk
  --work-base PATH      Temporary worktree directory on the data disk
  --direct-iters N      Direct decode iterations per cell (default: 50)
  --http-requests N     HTTP requests per concurrency unit (default: 5)
  --keep-worktrees      Keep clean worktrees after the run
  --shutdown-after      Shutdown only after a successful run
  -h, --help            Show this help
EOF
}

while (($# > 0)); do
    case "$1" in
        --model-4b) MODEL_4B=${2:?missing value for --model-4b}; shift 2 ;;
        --model-8b) MODEL_8B=${2:?missing value for --model-8b}; shift 2 ;;
        --gpu) GPU=${2:?missing value for --gpu}; shift 2 ;;
        --result-base) RESULT_BASE=${2:?missing value for --result-base}; shift 2 ;;
        --work-base) WORK_BASE=${2:?missing value for --work-base}; shift 2 ;;
        --direct-iters) DIRECT_ITERS=${2:?missing value for --direct-iters}; shift 2 ;;
        --http-requests) HTTP_REQUESTS_PER_CONCURRENCY=${2:?missing value for --http-requests}; shift 2 ;;
        --keep-worktrees) KEEP_WORKTREES=1; shift ;;
        --shutdown-after) SHUTDOWN_AFTER=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

test -d "$REPO_ROOT"
test -f "$MODEL_4B/config.json"
test -f "$MODEL_8B/config.json"
test "$DIRECT_ITERS" -gt 0
test "$HTTP_REQUESTS_PER_CONCURRENCY" -gt 0
test "$READY_TIMEOUT" -gt 0

cd "$REPO_ROOT"
REPO_ROOT=$(git rev-parse --show-toplevel)
test "$REPO_ROOT" = "/root/openinfer" || {
    echo "unexpected repository root: $REPO_ROOT" >&2
    exit 2
}

command -v git >/dev/null
command -v cargo >/dev/null
command -v curl >/dev/null
command -v nvidia-smi >/dev/null
command -v python3 >/dev/null
command -v sha256sum >/dev/null

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" | head -n 1)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$GPU" | head -n 1)
GPU_SM=${GPU_CC/./}
PEGAINFER_CUDA_SM=${PEGAINFER_CUDA_SM:-$GPU_SM}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU}

CANDIDATE_COMMIT=${CANDIDATE_COMMIT:-$(git rev-parse HEAD)}
git cat-file -e "${BASELINE_COMMIT}^{commit}"
git cat-file -e "${CANDIDATE_COMMIT}^{commit}"

STAMP=$(date +%Y%m%d-%H%M%S)-$$
RUN_ROOT=${RESULT_BASE}/qwen3-746-e2e-ab-${STAMP}
BASE_WT=${WORK_BASE}/${STAMP}-baseline
CAND_WT=${WORK_BASE}/${STAMP}-candidate
BASE_TARGET=${WORK_BASE}/${STAMP}-cargo-target-baseline
CAND_TARGET=${WORK_BASE}/${STAMP}-cargo-target-candidate
mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs" "$RUN_ROOT/direct" "$RUN_ROOT/http" \
    "$RUN_ROOT/meta" "$WORK_BASE"

SERVER_PID=
SERVER_LOG=
SERVER_PORT=$PORT
FINAL_STATUS=FAILED

run_logged() {
    local name=$1
    shift
    echo "===== ${name} ====="
    "$@" 2>&1 | tee "$RUN_ROOT/logs/${name}.log"
    local rc=${PIPESTATUS[0]}
    if ((rc != 0)); then
        echo "FAILED: ${name} rc=${rc}" >&2
        return "$rc"
    fi
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
    SERVER_LOG=
}

port_free() {
    python3 - "$SERVER_PORT" <<'PY'
import socket
import sys

sock = socket.socket()
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    sock.close()
PY
}

wait_ready() {
    local deadline=$((SECONDS + READY_TIMEOUT))
    while ((SECONDS < deadline)); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "server exited before readiness: ${SERVER_LOG}" >&2
            tail -n 120 "$SERVER_LOG" >&2 || true
            return 1
        fi
        if curl --fail --silent --max-time 2 \
            "http://127.0.0.1:${SERVER_PORT}/v1/models" \
            > "$RUN_ROOT/meta/models-${SERVER_PORT}.json"; then
            return 0
        fi
        sleep 1
    done
    echo "server readiness timeout: ${SERVER_LOG}" >&2
    tail -n 120 "$SERVER_LOG" >&2 || true
    return 1
}

start_server() {
    local revision=$1
    local model_label=$2
    local model_path=$3
    local served_model=$4
    local binary
    if [[ "$revision" == baseline ]]; then
        binary=${BASE_TARGET}/release/pegainfer
    else
        binary=${CAND_TARGET}/release/pegainfer
    fi
    test -x "$binary"
    stop_server
    port_free
    SERVER_LOG=${RUN_ROOT}/logs/server-${revision}-${model_label}.log
    echo "starting ${revision} ${model_label}: ${binary}"
    env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        PEGAINFER_CUDA_SM="$PEGAINFER_CUDA_SM" \
        RUST_LOG=info PEGAINFER_BASIC_HTTP_TRACE=1 \
        "$binary" \
        --model-path "$model_path" \
        --served-model-name "$served_model" \
        --port "$SERVER_PORT" \
        --tp-size 1 \
        --cuda-graph=true \
        --decode-overlap off \
        --no-prefix-cache \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    wait_ready
    if [[ "$revision" == candidate ]]; then
        grep -Fq 'decode projection path: resolved=FusedQkv' "$SERVER_LOG" || {
            echo "candidate did not resolve FusedQkv: ${SERVER_LOG}" >&2
            tail -n 120 "$SERVER_LOG" >&2 || true
            return 1
        }
    fi
}

run_http_cell() {
    local revision=$1
    local model_label=$2
    local model_path=$3
    local served_model=$4
    local profile=$5
    local concurrency=$6
    local order=$7
    local repeat=$8
    local prompt_words max_tokens requests binary commit out
    if [[ "$profile" == prefill ]]; then
        prompt_words=10000
        max_tokens=1
    else
        prompt_words=1024
        max_tokens=256
    fi
    requests=$((HTTP_REQUESTS_PER_CONCURRENCY * concurrency))
    if [[ "$revision" == baseline ]]; then
        binary=${BASE_TARGET}/release/pegainfer
        commit=$BASELINE_COMMIT
    else
        binary=${CAND_TARGET}/release/pegainfer
        commit=$CANDIDATE_COMMIT
    fi
    out=${RUN_ROOT}/http/order${order}-${revision}-${model_label}-${profile}-c${concurrency}-r${repeat}.json
    run_logged "http-order${order}-${revision}-${model_label}-${profile}-c${concurrency}" \
        python3 "$CAND_WT/scripts/bench_http_serving.py" \
        --base-url "http://127.0.0.1:${SERVER_PORT}" \
        --model "$served_model" \
        --model-path "$model_path" \
        --server-command "$binary --model-path $model_path --tp-size 1 --cuda-graph=true --no-prefix-cache" \
        --server-binary "$binary" \
        --commit "$commit" \
        --source-revision "$commit" \
        --backend pegainfer \
        --num-requests "$requests" \
        --concurrency "$concurrency" \
        --warmup "$concurrency" \
        --prompt-words "$prompt_words" \
        --max-tokens "$max_tokens" \
        --prompt-seed 746 \
        --temperature 0 \
        --ignore-eos \
        --timeout 900 \
        --server-log "$SERVER_LOG" \
        --out "$out"
}

run_direct_cell() {
    local revision=$1
    local model_label=$2
    local model_path=$3
    local graph=$4
    local batch_size=$5
    local order=$6
    local repeat=$7
    local binary graph_args=() out
    if [[ "$revision" == baseline ]]; then
        binary=${BASE_TARGET}/release/qwen3_decode_context
    else
        binary=${CAND_TARGET}/release/qwen3_decode_context
    fi
    if [[ "$graph" == eager ]]; then
        graph_args+=(--disable-cuda-graph)
    fi
    out=${RUN_ROOT}/direct/order${order}-${revision}-${model_label}-${graph}-bs${batch_size}-r${repeat}.log
    run_logged "direct-order${order}-${revision}-${model_label}-${graph}-bs${batch_size}" \
        env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        PEGAINFER_CUDA_SM="$PEGAINFER_CUDA_SM" \
        "$binary" --mode measure --model-path "$model_path" \
        --contexts 1024 --iters "$DIRECT_ITERS" --batch-size "$batch_size" \
        "${graph_args[@]}" > "$out"
}

summarize_results() {
    python3 - "$RUN_ROOT" <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
http = {}
pattern = re.compile(
    r"order(?P<order>\d+)-(?P<revision>baseline|candidate)-"
    r"(?P<model>qwen3-[48]b)-(?P<profile>prefill|decode)-c(?P<conc>\d+)-r(?P<repeat>\d+)\.json$"
)
for path in sorted((root / "http").glob("*.json")):
    match = pattern.match(path.name)
    if not match:
        continue
    data = json.loads(path.read_text())
    key = tuple(match.group(x) for x in ("model", "profile", "conc"))
    record = {
        "revision": match.group("revision"),
        "tpot_p50_ms": data.get("metrics", {}).get("tpot", {}).get("p50_ms"),
        "output_tokens_per_s": data.get("summary", {}).get("output_tokens_per_s"),
        "completed": data.get("summary", {}).get("completed"),
        "failed": data.get("summary", {}).get("failed"),
    }
    http.setdefault(key, {}).setdefault(record["revision"], []).append(record)

print("HTTP A/B summary")
for key, revisions in sorted(http.items()):
    model, profile, conc = key
    print(f"{model} {profile} concurrency={conc}")
    for revision in ("baseline", "candidate"):
        rows = revisions.get(revision, [])
        tpot = [r["tpot_p50_ms"] for r in rows if r["tpot_p50_ms"] is not None]
        out = [r["output_tokens_per_s"] for r in rows if r["output_tokens_per_s"] is not None]
        failed = sum((r["failed"] or 0) for r in rows)
        print(
            f"  {revision}: runs={len(rows)} failed={failed} "
            f"tpot_p50_ms={statistics.fmean(tpot) if tpot else 'NA'} "
            f"output_tok_s={statistics.fmean(out) if out else 'NA'}"
        )
    base = revisions.get("baseline", [])
    cand = revisions.get("candidate", [])
    base_t = [r["tpot_p50_ms"] for r in base if r["tpot_p50_ms"] is not None]
    cand_t = [r["tpot_p50_ms"] for r in cand if r["tpot_p50_ms"] is not None]
    base_o = [r["output_tokens_per_s"] for r in base if r["output_tokens_per_s"] is not None]
    cand_o = [r["output_tokens_per_s"] for r in cand if r["output_tokens_per_s"] is not None]
    if base_t and cand_t:
        bt, ct = statistics.fmean(base_t), statistics.fmean(cand_t)
        print(f"  delta: tpot={(bt-ct)/bt*100.0:+.3f}%")
    if base_o and cand_o:
        bo, co = statistics.fmean(base_o), statistics.fmean(cand_o)
        print(f"  delta: output_throughput={(co-bo)/bo*100.0:+.3f}%")

direct_pattern = re.compile(
    r"order(?P<order>\d+)-(?P<revision>baseline|candidate)-"
    r"(?P<model>qwen3-[48]b)-(?P<graph>graph|eager)-bs(?P<bs>\d+)-r(?P<repeat>\d+)\.log$"
)
direct = {}
for path in sorted((root / "direct").glob("*.log")):
    match = direct_pattern.match(path.name)
    if not match:
        continue
    values = []
    for line in path.read_text().splitlines():
        if line.startswith("1024,"):
            fields = line.split(",")
            if len(fields) >= 8:
                values.append(float(fields[4]))
    key = tuple(match.group(x) for x in ("model", "graph", "bs"))
    direct.setdefault(key, {}).setdefault(match.group("revision"), []).extend(values)

print("Direct A/B summary (avg_ms)")
for key, revisions in sorted(direct.items()):
    print(" ".join(key))
    for revision in ("baseline", "candidate"):
        values = revisions.get(revision, [])
        print(f"  {revision}: samples={len(values)} avg_ms={statistics.fmean(values) if values else 'NA'}")
    base = revisions.get("baseline", [])
    cand = revisions.get("candidate", [])
    if base and cand:
        print(f"  delta: {(statistics.fmean(base)-statistics.fmean(cand))/statistics.fmean(base)*100.0:+.3f}%")
PY
}

cleanup() {
    local rc=$?
    stop_server || true
    if [[ "$KEEP_WORKTREES" != 1 ]]; then
        git worktree remove --force "$BASE_WT" 2>/dev/null || true
        git worktree remove --force "$CAND_WT" 2>/dev/null || true
    fi
    if ((rc == 0)); then
        FINAL_STATUS=PASSED
    fi
    printf '%s\n' "$FINAL_STATUS" > "$RUN_ROOT/runner-status.txt"
    local archive=${RESULT_BASE}/$(basename "$RUN_ROOT").tar.gz
    local sha_file=${archive}.sha256
    if tar -czf "$archive" -C "$RESULT_BASE" "$(basename "$RUN_ROOT")"; then
        (cd "$(dirname "$archive")" && sha256sum "$(basename "$archive")") > "$sha_file"
    else
        echo "evidence archive failed" >&2
    fi
    if [[ "$BASE_TARGET" == "$WORK_BASE"/* && -d "$BASE_TARGET" ]]; then
        rm -rf -- "$BASE_TARGET"
    fi
    if [[ "$CAND_TARGET" == "$WORK_BASE"/* && -d "$CAND_TARGET" ]]; then
        rm -rf -- "$CAND_TARGET"
    fi
    if [[ "$FINAL_STATUS" == PASSED ]]; then
        echo "FINAL_STATUS=PASSED"
        echo "RUN_ROOT=$RUN_ROOT"
        echo "ARCHIVE=$archive"
        echo "SHA256=$sha_file"
        if [[ "$SHUTDOWN_AFTER" == 1 ]]; then
            /usr/bin/shutdown
        fi
    else
        echo "FINAL_STATUS=FAILED" >&2
        echo "RUN_ROOT=$RUN_ROOT" >&2
        echo "ARCHIVE=$archive" >&2
        echo "SHA256=$sha_file" >&2
    fi
    exit "$rc"
}
trap cleanup EXIT

{
    echo "repo=$REPO_ROOT"
    echo "baseline_commit=$BASELINE_COMMIT"
    echo "candidate_commit=$CANDIDATE_COMMIT"
    echo "gpu=$GPU_NAME"
    echo "compute_capability=$GPU_CC"
    echo "PEGAINFER_CUDA_SM=$PEGAINFER_CUDA_SM"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "rustc=$(rustc --version)"
    echo "cargo=$(cargo --version)"
    nvidia-smi -L
} > "$RUN_ROOT/meta/environment.txt"
git -C "$REPO_ROOT" status --short --untracked-files=all > "$RUN_ROOT/meta/source-checkout-status.txt"

run_logged worktree-baseline git worktree add --detach "$BASE_WT" "$BASELINE_COMMIT"
run_logged worktree-candidate git worktree add --detach "$CAND_WT" "$CANDIDATE_COMMIT"
run_logged submodules-baseline git -C "$BASE_WT" submodule update --init --recursive
run_logged submodules-candidate git -C "$CAND_WT" submodule update --init --recursive

git -C "$BASE_WT" status --short --untracked-files=no > "$RUN_ROOT/meta/baseline-tracked-status.txt"
git -C "$CAND_WT" status --short --untracked-files=no > "$RUN_ROOT/meta/candidate-tracked-status.txt"
test ! -s "$RUN_ROOT/meta/baseline-tracked-status.txt"
test ! -s "$RUN_ROOT/meta/candidate-tracked-status.txt"

run_logged build-baseline env CARGO_TARGET_DIR="$BASE_TARGET" \
    PEGAINFER_CUDA_SM="$PEGAINFER_CUDA_SM" \
    cargo build --release --locked --manifest-path "$BASE_WT/Cargo.toml" \
    -p pegainfer-server -p pegainfer-qwen3 --bin pegainfer --bin qwen3_decode_context
run_logged build-candidate env CARGO_TARGET_DIR="$CAND_TARGET" \
    PEGAINFER_CUDA_SM="$PEGAINFER_CUDA_SM" \
    cargo build --release --locked --manifest-path "$CAND_WT/Cargo.toml" \
    -p pegainfer-server -p pegainfer-qwen3 --bin pegainfer --bin qwen3_decode_context

MODEL_LABELS=(qwen3-4b qwen3-8b)
MODEL_PATHS=("$MODEL_4B" "$MODEL_8B")
SERVED_MODELS=(Qwen3-4B Qwen3-8B)
ORDER_REVISIONS=(baseline candidate candidate baseline)

for model_index in 0 1; do
    model_label=${MODEL_LABELS[$model_index]}
    model_path=${MODEL_PATHS[$model_index]}
    served_model=${SERVED_MODELS[$model_index]}
    for order_index in 0 1 2 3; do
        revision=${ORDER_REVISIONS[$order_index]}
        repeat=$((order_index / 2 + 1))
        start_server "$revision" "$model_label" "$model_path" "$served_model"
        for profile in prefill decode; do
            for concurrency in 1 8; do
                run_http_cell "$revision" "$model_label" "$model_path" \
                    "$served_model" "$profile" "$concurrency" \
                    "$((order_index + 1))" "$repeat"
            done
        done
        stop_server
    done
done

for model_index in 0 1; do
    model_label=${MODEL_LABELS[$model_index]}
    model_path=${MODEL_PATHS[$model_index]}
    for graph in graph eager; do
        for batch_size in 1 8; do
            for order_index in 0 1 2 3; do
                revision=${ORDER_REVISIONS[$order_index]}
                repeat=$((order_index / 2 + 1))
                run_direct_cell "$revision" "$model_label" "$model_path" \
                    "$graph" "$batch_size" "$((order_index + 1))" "$repeat"
            done
        done
    done
done

summarize_results | tee "$RUN_ROOT/summary.txt"
printf '%s\n' "source_commit=$CANDIDATE_COMMIT" > "$RUN_ROOT/meta/decision-input.txt"
printf '%s\n' "baseline_commit=$BASELINE_COMMIT" >> "$RUN_ROOT/meta/decision-input.txt"
printf '%s\n' "component_report=provided separately by qwen3-746-projection-ab" >> "$RUN_ROOT/meta/decision-input.txt"
