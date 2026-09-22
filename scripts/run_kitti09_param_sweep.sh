#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

RUNS=15
TAG="kitti09_param"
BAG_DIR="/media/analeticiaromero/ExtremeSSD/ANA/KITTI/09_oficial"
BAG_FILE="09.bag"
ROS_DISTRO_NAME="${ROS_DISTRO:-melodic}"
SDV_WS="$ROOT_DIR"
SDV_LAUNCH="run_kitti_09_multicam_param.launch"
OUTPUT_DIR="$ROOT_DIR/src/sdv_loam/output/kitti-09_param_sweep"
ARCHIVE_DIR=""
SDV_WARMUP_SEC=5
POST_BAG_WAIT_SEC=5
BAG_DELAY_SEC=1.0
BAG_RATE=1.0
TIMEOUT_SEC=0
WAIT_PROGRESS_SEC=30
CONFIGS="margin_low,margin_high,switch_stable,rmse_strict,warmup_long"
DRY_RUN=0

ROSCORE_PID=""
SDV_PID=""
BAG_PID=""

usage() {
    cat <<EOF
Usage:
  $0 [options]

Options:
  --runs N              Number of repeated runs per config. Default: $RUNS
  --tag NAME            Output tag used in pred_<tag>_<config>_runNN.*. Default: $TAG
  --configs LIST        Comma-separated configs. Default: $CONFIGS
                        Available: base, margin_low, margin_high, switch_stable,
                                   rmse_strict, rmse_permissive, warmup_long
  --bag-dir DIR         Directory containing the bag. Default: $BAG_DIR
  --bag-file FILE       Bag filename. Default: $BAG_FILE
  --bag PATH            Full bag path. Overrides --bag-dir/--bag-file.
  --bag-delay SEC       rosbag play delay (-d). Default: $BAG_DELAY_SEC
  --bag-rate R          rosbag play rate (-r). Default: $BAG_RATE
  --sdv-ws DIR          SDV-LOAM multicam workspace. Default: $SDV_WS
  --sdv-launch FILE     SDV-LOAM launch file. Default: $SDV_LAUNCH
  --output-dir DIR      Directory where SDV-LOAM writes pred.*. Default: $OUTPUT_DIR
  --archive-dir DIR     Directory for archived run outputs. Default: <output-dir>/runs
  --warmup-sec SEC      Seconds to wait after starting SDV before bag. Default: $SDV_WARMUP_SEC
  --post-wait-sec SEC   Seconds to wait after bag exits before stopping SDV. Default: $POST_BAG_WAIT_SEC
  --timeout-sec SEC     Max seconds to wait for rosbag play per run. 0 disables. Default: $TIMEOUT_SEC
  --progress-sec SEC    Print wait progress every SEC seconds. 0 disables. Default: $WAIT_PROGRESS_SEC
  --ros-distro NAME     ROS distro setup to source. Default: $ROS_DISTRO_NAME
  --dry-run             Print configuration without executing.
  -h, --help            Show this help.

Examples:
  $0
  $0 --configs base,margin_low,margin_high --runs 15
  $0 --configs base --runs 3 --bag /media/analeticiaromero/ExtremeSSD/ANA/KITTI/09_oficial/09.bag
EOF
}

FULL_BAG_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --configs) CONFIGS="$2"; shift 2 ;;
        --bag-dir) BAG_DIR="$2"; shift 2 ;;
        --bag-file) BAG_FILE="$2"; shift 2 ;;
        --bag) FULL_BAG_PATH="$2"; shift 2 ;;
        --bag-delay) BAG_DELAY_SEC="$2"; shift 2 ;;
        --bag-rate) BAG_RATE="$2"; shift 2 ;;
        --sdv-ws) SDV_WS="$2"; shift 2 ;;
        --sdv-launch) SDV_LAUNCH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --archive-dir) ARCHIVE_DIR="$2"; shift 2 ;;
        --warmup-sec) SDV_WARMUP_SEC="$2"; shift 2 ;;
        --post-wait-sec) POST_BAG_WAIT_SEC="$2"; shift 2 ;;
        --timeout-sec) TIMEOUT_SEC="$2"; shift 2 ;;
        --progress-sec) WAIT_PROGRESS_SEC="$2"; shift 2 ;;
        --ros-distro) ROS_DISTRO_NAME="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
SDV_SETUP="$SDV_WS/devel/setup.bash"
if [[ -z "$ARCHIVE_DIR" ]]; then
    ARCHIVE_DIR="$OUTPUT_DIR/runs"
fi
if [[ -z "$FULL_BAG_PATH" ]]; then
    FULL_BAG_PATH="$BAG_DIR/$BAG_FILE"
fi

require_path() {
    local path="$1"
    local description="$2"
    if [[ ! -e "$path" ]]; then
        echo "Missing $description: $path" >&2
        exit 1
    fi
}

run_cmd() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '[dry-run] %q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

cleanup_process_group() {
    local pid="$1"
    local label="$2"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping $label (pid=$pid)"
        kill -INT "-$pid" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$pid" 2>/dev/null || return 0
            sleep 0.5
        done
        kill -TERM "-$pid" 2>/dev/null || true
    fi
}

cleanup() {
    cleanup_process_group "$BAG_PID" "rosbag play"
    cleanup_process_group "$SDV_PID" "SDV-LOAM"
    cleanup_process_group "$ROSCORE_PID" "roscore"
}
trap cleanup EXIT INT TERM

wait_for_ros_master() {
    local max_wait=30
    for _ in $(seq 1 "$max_wait"); do
        if bash -lc "source '$ROS_SETUP' && rostopic list >/dev/null 2>&1"; then
            return 0
        fi
        sleep 1
    done
    echo "ROS master did not become available within ${max_wait}s" >&2
    return 1
}

wait_with_timeout() {
    local pid="$1"
    local timeout="$2"
    local start
    local last_progress
    start="$(date +%s)"
    last_progress="$start"

    if [[ "$timeout" -le 0 ]]; then
        wait "$pid"
        return $?
    fi

    while kill -0 "$pid" 2>/dev/null; do
        local now
        now="$(date +%s)"
        if (( now - start >= timeout )); then
            echo "Timeout waiting for process pid=$pid after ${timeout}s" >&2
            return 124
        fi
        if [[ "$WAIT_PROGRESS_SEC" -gt 0 ]] && (( now - last_progress >= WAIT_PROGRESS_SEC )); then
            echo "Waiting for bag pid=$pid: $((now - start))s elapsed / ${timeout}s timeout"
            last_progress="$now"
        fi
        sleep 1
    done
    wait "$pid" || return $?
}

config_values() {
    local config="$1"
    case "$config" in
        base) echo "1000.0 3 12.0 4" ;;
        margin_low) echo "500.0 3 12.0 4" ;;
        margin_high) echo "1500.0 3 12.0 4" ;;
        switch_stable) echo "1000.0 5 12.0 4" ;;
        rmse_strict) echo "1000.0 3 9.0 4" ;;
        rmse_permissive) echo "1000.0 3 15.0 4" ;;
        warmup_long) echo "1000.0 3 12.0 6" ;;
        *)
            echo "Unknown config: $config" >&2
            echo "Available: base, margin_low, margin_high, switch_stable, rmse_strict, rmse_permissive, warmup_long" >&2
            return 1
            ;;
    esac
}

clear_default_outputs() {
    local files=(
        "$OUTPUT_DIR/pred.txt"
        "$OUTPUT_DIR/pred.tum"
        "$OUTPUT_DIR/pred_timestamps.txt"
        "$OUTPUT_DIR/pred_active_camera.txt"
        "$OUTPUT_DIR/pred_camera_scores.csv"
        "$OUTPUT_DIR/pred_timing.txt"
    )
    run_cmd rm -f "${files[@]}"
}

archive_outputs() {
    local config="$1"
    local run_id="$2"
    local run_label
    run_label="$(printf 'run%02d' "$run_id")"
    local prefix="$ARCHIVE_DIR/pred_${TAG}_${config}_${run_label}"

    mkdir -p "$ARCHIVE_DIR"

    local copied=0
    local src dst
    for src in \
        "$OUTPUT_DIR/pred.txt" \
        "$OUTPUT_DIR/pred.tum" \
        "$OUTPUT_DIR/pred_timestamps.txt" \
        "$OUTPUT_DIR/pred_active_camera.txt" \
        "$OUTPUT_DIR/pred_camera_scores.csv" \
        "$OUTPUT_DIR/pred_timing.txt"
    do
        if [[ ! -s "$src" ]]; then
            echo "Warning: missing or empty output for config ${config} run ${run_id}: $src" >&2
            continue
        fi

        case "$(basename "$src")" in
            pred.txt) dst="${prefix}.txt" ;;
            pred.tum) dst="${prefix}.tum" ;;
            pred_timestamps.txt) dst="${prefix}_timestamps.txt" ;;
            pred_active_camera.txt) dst="${prefix}_active_camera.txt" ;;
            pred_camera_scores.csv) dst="${prefix}_camera_scores.csv" ;;
            pred_timing.txt) dst="${prefix}_timing.txt" ;;
            *) dst="${prefix}_$(basename "$src")" ;;
        esac

        run_cmd cp -f "$src" "$dst"
        copied=$((copied + 1))
    done

    if [[ "$copied" -eq 0 ]]; then
        echo "No outputs were archived for config ${config} run ${run_id}" >&2
        return 1
    fi

    echo "Archived ${copied} output files with prefix: $prefix"
}

start_roscore() {
    if bash -lc "source '$ROS_SETUP' && rostopic list >/dev/null 2>&1"; then
        echo "Using existing ROS master"
        ROSCORE_PID=""
        return 0
    fi

    echo "Starting roscore"
    setsid bash -lc "source '$ROS_SETUP' && roscore" >"$ARCHIVE_DIR/roscore.log" 2>&1 &
    ROSCORE_PID=$!
    wait_for_ros_master
}

start_sdv() {
    local config="$1"
    local run_id="$2"
    local margin="$3"
    local consecutive="$4"
    local rmse="$5"
    local warmup="$6"
    local log_file="$ARCHIVE_DIR/sdv_${TAG}_${config}_run$(printf '%02d' "$run_id").log"

    echo "Starting SDV-LOAM config=${config} run=${run_id}"
    mkdir -p "$OUTPUT_DIR"
    setsid bash -lc "source '$ROS_SETUP' && source '$SDV_SETUP' && roslaunch sdv_loam '$SDV_LAUNCH' resultPath:='$OUTPUT_DIR/pred.txt' cameraSwitchScoreMargin:='$margin' cameraSwitchConsecutiveFrames:='$consecutive' maxCoarseTrackingRMSE:='$rmse' minFallbackWarmupFrames:='$warmup'" >"$log_file" 2>&1 &
    SDV_PID=$!
}

start_bag() {
    local config="$1"
    local run_id="$2"
    local log_file="$ARCHIVE_DIR/bag_${TAG}_${config}_run$(printf '%02d' "$run_id").log"
    echo "Starting rosbag play config=${config} run=${run_id}"
    setsid bash -lc "source '$ROS_SETUP' && rosbag play '$FULL_BAG_PATH' --clock -d '$BAG_DELAY_SEC' -r '$BAG_RATE'" >"$log_file" 2>&1 &
    BAG_PID=$!
}

require_path "$ROS_SETUP" "ROS setup"
require_path "$SDV_SETUP" "SDV-LOAM multicam setup"
require_path "$FULL_BAG_PATH" "input bag"

IFS=',' read -r -a CONFIG_ARRAY <<< "$CONFIGS"
for config in "${CONFIG_ARRAY[@]}"; do
    config="${config//[[:space:]]/}"
    config_values "$config" >/dev/null
done

mkdir -p "$ARCHIVE_DIR"

echo "Configuration:"
echo "  runs/config: $RUNS"
echo "  tag:         $TAG"
echo "  configs:     $CONFIGS"
echo "  bag:         $FULL_BAG_PATH"
echo "  bag delay:   $BAG_DELAY_SEC"
echo "  bag rate:    $BAG_RATE"
echo "  timeout:     ${TIMEOUT_SEC}s"
echo "  progress:    every ${WAIT_PROGRESS_SEC}s"
echo "  output dir:  $OUTPUT_DIR"
echo "  archive dir: $ARCHIVE_DIR"
echo "  launch:      $SDV_LAUNCH"
echo
echo "Config values: cameraSwitchScoreMargin cameraSwitchConsecutiveFrames maxCoarseTrackingRMSE minFallbackWarmupFrames"
for config in "${CONFIG_ARRAY[@]}"; do
    config="${config//[[:space:]]/}"
    echo "  $config: $(config_values "$config")"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry-run mode: no commands will be executed."
    exit 0
fi

start_roscore

for config in "${CONFIG_ARRAY[@]}"; do
    config="${config//[[:space:]]/}"
    read -r margin consecutive rmse warmup <<< "$(config_values "$config")"

    for run_id in $(seq 1 "$RUNS"); do
        echo
        echo "========== KITTI 09 param sweep ${config} run ${run_id}/${RUNS} =========="
        SDV_PID=""
        BAG_PID=""

        clear_default_outputs
        start_sdv "$config" "$run_id" "$margin" "$consecutive" "$rmse" "$warmup"
        sleep "$SDV_WARMUP_SEC"
        start_bag "$config" "$run_id"

        bag_status=0
        wait_with_timeout "$BAG_PID" "$TIMEOUT_SEC" || bag_status=$?

        if [[ "$bag_status" -eq 124 ]]; then
            echo "rosbag play reached timeout (${TIMEOUT_SEC}s) in config ${config} run ${run_id}; stopping bag." >&2
            cleanup_process_group "$BAG_PID" "rosbag play"
        elif [[ "$bag_status" -ne 0 ]]; then
            echo "rosbag play exited with status ${bag_status} in config ${config} run ${run_id}." >&2
        fi
        BAG_PID=""

        sleep "$POST_BAG_WAIT_SEC"
        cleanup_process_group "$SDV_PID" "SDV-LOAM"
        wait "$SDV_PID" 2>/dev/null || true
        SDV_PID=""

        archive_outputs "$config" "$run_id"
    done
done

echo
echo "All parameter sweep runs finished. Outputs are in: $ARCHIVE_DIR"
