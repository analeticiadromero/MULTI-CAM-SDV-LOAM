#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

RUNS=20
TAG="kitti360"
DATASET_DIR="/media/analeticiaromero/ExtremeSSD/ANA/kitti-360/KITTI-360"
SEQUENCE=4
START_TIME=12.86
END_TIME=""
RATE=1
ROS_DISTRO_NAME="${ROS_DISTRO:-melodic}"
SDV_WS="$ROOT_DIR"
PLAYER_WS="/home/analeticiaromero/kitti360_ros_player_ws"
SDV_LAUNCH="run_kitti_360_multicam.launch"
OUTPUT_DIR="$ROOT_DIR/src/sdv_loam/output/kitti-360"
ARCHIVE_DIR=""
SDV_WARMUP_SEC=5
POST_PLAYER_WAIT_SEC=5
TIMEOUT_SEC=0
AUTO_TIMEOUT_MARGIN_SEC=20
WAIT_PROGRESS_SEC=30
DRY_RUN=0

ROSCORE_PID=""
SDV_PID=""
PLAYER_PID=""

usage() {
    cat <<EOF
Usage:
  $0 [options]

Options:
  --runs N              Number of repeated runs. Default: $RUNS
  --tag NAME            Output tag used in pred_<tag>_runNN.*. Default: $TAG
  --dataset DIR         KITTI-360 dataset directory. Default: $DATASET_DIR
  --sequence N          KITTI-360 sequence. Default: $SEQUENCE
  --start SEC           Player start time. Default: $START_TIME
  --end SEC             Optional player end time.
  --rate R              Player playback rate. Default: $RATE
  --sdv-ws DIR          SDV-LOAM workspace. Default: $SDV_WS
  --player-ws DIR       KITTI-360 player workspace. Default: $PLAYER_WS
  --sdv-launch FILE     SDV-LOAM launch file. Default: $SDV_LAUNCH
  --output-dir DIR      Directory where SDV-LOAM writes pred.*. Default: $OUTPUT_DIR
  --archive-dir DIR     Directory for archived run outputs. Default: <output-dir>/runs
  --warmup-sec SEC      Seconds to wait after starting SDV before player. Default: $SDV_WARMUP_SEC
  --post-wait-sec SEC   Seconds to wait after player exits before stopping SDV. Default: $POST_PLAYER_WAIT_SEC
  --timeout-sec SEC     Max seconds to wait for player per run. 0 disables. Default: $TIMEOUT_SEC
  --auto-timeout-margin-sec SEC
                        Extra wall-clock seconds added to auto timeout when --end is set. Default: $AUTO_TIMEOUT_MARGIN_SEC
  --progress-sec SEC    Print wait progress every SEC seconds. 0 disables. Default: $WAIT_PROGRESS_SEC
  --ros-distro NAME     ROS distro setup to source. Default: $ROS_DISTRO_NAME
  --dry-run             Print commands without executing.
  -h, --help            Show this help.

Example:
  $0 --runs 20 --tag original --dataset /media/analeticiaromero/ExtremeSSD/ANA/kitti-360/KITTI-360 --sequence 4 --start 12.86
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --dataset) DATASET_DIR="$2"; shift 2 ;;
        --sequence) SEQUENCE="$2"; shift 2 ;;
        --start) START_TIME="$2"; shift 2 ;;
        --end) END_TIME="$2"; shift 2 ;;
        --rate) RATE="$2"; shift 2 ;;
        --sdv-ws) SDV_WS="$2"; shift 2 ;;
        --player-ws) PLAYER_WS="$2"; shift 2 ;;
        --sdv-launch) SDV_LAUNCH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --archive-dir) ARCHIVE_DIR="$2"; shift 2 ;;
        --warmup-sec) SDV_WARMUP_SEC="$2"; shift 2 ;;
        --post-wait-sec) POST_PLAYER_WAIT_SEC="$2"; shift 2 ;;
        --timeout-sec) TIMEOUT_SEC="$2"; shift 2 ;;
        --auto-timeout-margin-sec) AUTO_TIMEOUT_MARGIN_SEC="$2"; shift 2 ;;
        --progress-sec) WAIT_PROGRESS_SEC="$2"; shift 2 ;;
        --ros-distro) ROS_DISTRO_NAME="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
SDV_SETUP="$SDV_WS/devel/setup.bash"
PLAYER_SETUP="$PLAYER_WS/devel/setup.bash"
if [[ -z "$ARCHIVE_DIR" ]]; then
    ARCHIVE_DIR="$OUTPUT_DIR/runs"
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
    cleanup_process_group "$PLAYER_PID" "KITTI-360 player"
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
            echo "Waiting for player pid=$pid: $((now - start))s elapsed / ${timeout}s timeout"
            last_progress="$now"
        fi
        sleep 1
    done
    wait "$pid" || return $?
}

get_effective_player_timeout() {
    if [[ "$TIMEOUT_SEC" != "0" ]]; then
        echo "$TIMEOUT_SEC"
        return 0
    fi

    if [[ -z "$END_TIME" ]]; then
        echo "0"
        return 0
    fi

    python3 - "$START_TIME" "$END_TIME" "$RATE" "$AUTO_TIMEOUT_MARGIN_SEC" <<'PY'
import math
import sys

start = float(sys.argv[1])
end = float(sys.argv[2])
rate = float(sys.argv[3])
margin = float(sys.argv[4])

if rate <= 0:
    raise SystemExit("rate must be > 0 to compute automatic timeout")

duration = max(0.0, (end - start) / rate)
print(int(math.ceil(duration + margin)))
PY
}

clear_default_outputs() {
    local files=(
        "$OUTPUT_DIR/pred.txt"
        "$OUTPUT_DIR/pred.tum"
        "$OUTPUT_DIR/pred_timestamps.txt"
        "$OUTPUT_DIR/pred_active_camera.txt"
        "$OUTPUT_DIR/pred_camera_scores.csv"
    )
    run_cmd rm -f "${files[@]}"
}

archive_outputs() {
    local run_id="$1"
    local run_label
    run_label="$(printf 'run%02d' "$run_id")"
    local prefix="$ARCHIVE_DIR/pred_${TAG}_${run_label}"

    mkdir -p "$ARCHIVE_DIR"

    local copied=0
    local src dst
    for src in \
        "$OUTPUT_DIR/pred.txt" \
        "$OUTPUT_DIR/pred.tum" \
        "$OUTPUT_DIR/pred_timestamps.txt" \
        "$OUTPUT_DIR/pred_active_camera.txt" \
        "$OUTPUT_DIR/pred_camera_scores.csv"
    do
        if [[ ! -s "$src" ]]; then
            echo "Warning: missing or empty output for run ${run_id}: $src" >&2
            continue
        fi

        case "$(basename "$src")" in
            pred.txt) dst="${prefix}.txt" ;;
            pred.tum) dst="${prefix}.tum" ;;
            pred_timestamps.txt) dst="${prefix}_timestamps.txt" ;;
            pred_active_camera.txt) dst="${prefix}_active_camera.txt" ;;
            pred_camera_scores.csv) dst="${prefix}_camera_scores.csv" ;;
            *) dst="${prefix}_$(basename "$src")" ;;
        esac

        run_cmd cp -f "$src" "$dst"
        copied=$((copied + 1))
    done

    if [[ "$copied" -eq 0 ]]; then
        echo "No outputs were archived for run ${run_id}" >&2
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
    local run_id="$1"
    local log_file="$ARCHIVE_DIR/sdv_${TAG}_run$(printf '%02d' "$run_id").log"
    echo "Starting SDV-LOAM for run ${run_id}"
    mkdir -p "$OUTPUT_DIR"
    setsid bash -lc "source '$ROS_SETUP' && source '$SDV_SETUP' && roslaunch sdv_loam '$SDV_LAUNCH' resultPath:='$OUTPUT_DIR/pred.txt'" >"$log_file" 2>&1 &
    SDV_PID=$!
}

start_player() {
    local run_id="$1"
    local log_file="$ARCHIVE_DIR/player_${TAG}_run$(printf '%02d' "$run_id").log"
    local end_arg=""
    if [[ -n "$END_TIME" ]]; then
        end_arg="end:=$END_TIME"
    fi

    echo "Starting KITTI-360 player for run ${run_id}"
    setsid bash -lc "source '$ROS_SETUP' && source '$PLAYER_SETUP' && roslaunch kitti360_publisher Kitti360.launch \
        directory:='$DATASET_DIR' \
        sequence:='$SEQUENCE' \
        start:='$START_TIME' \
        rate:='$RATE' \
        looping:=False \
        pub_velodyne:=True \
        pub_perspective_rectified_left:=True \
        pub_perspective_rectified_right:=True \
        pub_velodyne_labeled:=False \
        pub_sick_points:=False \
        pub_perspective_unrectified_left:=False \
        pub_perspective_unrectified_right:=False \
        pub_fisheye_left:=False \
        pub_fisheye_right:=False \
        pub_bounding_boxes:=False \
        pub_bounding_boxes_rviz_marker:=False \
        pub_2d_semantics_left:=False \
        pub_2d_semantics_right:=False \
        pub_2d_semantics_rgb_left:=False \
        pub_2d_semantics_rgb_right:=False \
        pub_2d_instance_left:=False \
        pub_2d_instance_right:=False \
        pub_2d_confidence_left:=False \
        pub_2d_confidence_right:=False \
        pub_3d_semantics_static:=False \
        pub_3d_semantics_dynamic:=False \
        pub_camera_intrinsics:=True \
        $end_arg" >"$log_file" 2>&1 &
    PLAYER_PID=$!
}

require_path "$ROS_SETUP" "ROS setup"
require_path "$SDV_SETUP" "SDV-LOAM setup"
require_path "$PLAYER_SETUP" "KITTI-360 player setup"
require_path "$DATASET_DIR" "KITTI-360 dataset directory"

mkdir -p "$ARCHIVE_DIR"

echo "Configuration:"
echo "  runs:        $RUNS"
echo "  tag:         $TAG"
echo "  dataset:     $DATASET_DIR"
echo "  sequence:    $SEQUENCE"
echo "  start:       $START_TIME"
echo "  end:         ${END_TIME:-<none>}"
echo "  timeout:     $(get_effective_player_timeout)s"
echo "  progress:    every ${WAIT_PROGRESS_SEC}s"
echo "  output dir:  $OUTPUT_DIR"
echo "  archive dir: $ARCHIVE_DIR"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry-run mode: no commands will be executed."
    exit 0
fi

start_roscore

for run_id in $(seq 1 "$RUNS"); do
    echo
    echo "========== KITTI-360 run ${run_id}/${RUNS} =========="
    SDV_PID=""
    PLAYER_PID=""

    clear_default_outputs
    start_sdv "$run_id"
    sleep "$SDV_WARMUP_SEC"
    start_player "$run_id"

    player_status=0
    effective_timeout="$(get_effective_player_timeout)"
    wait_with_timeout "$PLAYER_PID" "$effective_timeout" || player_status=$?

    if [[ "$player_status" -eq 124 ]]; then
        echo "Player reached timeout (${effective_timeout}s) in run ${run_id}; stopping player." >&2
        cleanup_process_group "$PLAYER_PID" "KITTI-360 player"
    elif [[ "$player_status" -ne 0 ]]; then
        echo "Player exited with status ${player_status} in run ${run_id}." >&2
    fi
    PLAYER_PID=""

    sleep "$POST_PLAYER_WAIT_SEC"
    cleanup_process_group "$SDV_PID" "SDV-LOAM"
    wait "$SDV_PID" 2>/dev/null || true
    SDV_PID=""

    archive_outputs "$run_id"
done

echo
echo "All runs finished. Outputs are in: $ARCHIVE_DIR"
