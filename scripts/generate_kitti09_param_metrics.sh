#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

GT_FILE="/home/analeticiaromero/09.txt"
RESULTS_DIR="$ROOT_DIR/src/sdv_loam/output/kitti-09_param_sweep/runs"
METRICS_DIR="/home/analeticiaromero/sdv_loam_results/kitti-09/param_sweep"
TAG="kitti09_param"
RUNS=15
CONFIGS="margin_low,margin_high,switch_stable,rmse_strict,warmup_long"
ALIGN="-a"
CLEAN=0
DRY_RUN=0

usage() {
    cat <<EOF
Usage:
  $0 [options]

Options:
  --gt FILE             KITTI-format ground truth. Default: $GT_FILE
  --results-dir DIR     Directory with pred_<tag>_<config>_runNN.txt. Default: $RESULTS_DIR
  --metrics-dir DIR     Directory for EVO ZIPs and CSV tables. Default: $METRICS_DIR
  --tag NAME            Prediction tag prefix. Default: $TAG
  --runs N              Number of runs per config. Default: $RUNS
  --configs LIST        Comma-separated configs. Default: $CONFIGS
  --no-align            Do not pass -a to evo_ape/evo_rpe.
  --clean               Remove previous ape/rpe ZIPs and CSVs in metrics dir before running.
  --dry-run             Print commands without executing EVO.
  -h, --help            Show this help.

Examples:
  $0
  $0 --configs margin_low,margin_high --runs 15
  $0 --gt /home/analeticiaromero/devkit_odometry/cpp/data/poses/09.txt
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gt) GT_FILE="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --metrics-dir) METRICS_DIR="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --configs) CONFIGS="$2"; shift 2 ;;
        --no-align) ALIGN=""; shift ;;
        --clean) CLEAN=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

require_file() {
    local path="$1"
    local description="$2"
    if [[ ! -f "$path" ]]; then
        echo "Missing $description: $path" >&2
        exit 1
    fi
}

require_dir() {
    local path="$1"
    local description="$2"
    if [[ ! -d "$path" ]]; then
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

require_file "$GT_FILE" "KITTI ground truth"
require_dir "$RESULTS_DIR" "results directory"

IFS=',' read -r -a CONFIG_ARRAY <<< "$CONFIGS"

echo "=================================================="
echo " EXTRACAO DE METRICAS EVO - KITTI 09 PARAM SWEEP"
echo "=================================================="
echo "  gt:          $GT_FILE"
echo "  results dir: $RESULTS_DIR"
echo "  metrics dir: $METRICS_DIR"
echo "  tag:         $TAG"
echo "  configs:     $CONFIGS"
echo "  runs/config: $RUNS"
echo "  alignment:   ${ALIGN:-disabled}"

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$METRICS_DIR"
fi

if [[ "$CLEAN" -eq 1 ]]; then
    echo "Cleaning previous metrics in: $METRICS_DIR"
    run_cmd rm -f \
        "$METRICS_DIR"/ape_*.zip \
        "$METRICS_DIR"/rpe_*.zip \
        "$METRICS_DIR"/tabela_resumo_*.csv
fi

found=0
for config in "${CONFIG_ARRAY[@]}"; do
    config="${config//[[:space:]]/}"
    config_dir="$METRICS_DIR/$config"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        mkdir -p "$config_dir"
    fi

    echo
    echo ">> Processando configuracao: $config"

    for run in $(seq 1 "$RUNS"); do
        run_id="$(printf '%02d' "$run")"
        est_file="$RESULTS_DIR/pred_${TAG}_${config}_run${run_id}.txt"

        if [[ ! -f "$est_file" ]]; then
            echo "   [AVISO] Arquivo nao encontrado: $est_file"
            continue
        fi

        found=$((found + 1))
        ape_zip="$config_dir/ape_${TAG}_${config}_run${run_id}.zip"
        rpe_zip="$config_dir/rpe_${TAG}_${config}_run${run_id}.zip"

        if [[ -n "$ALIGN" ]]; then
            run_cmd evo_ape kitti "$GT_FILE" "$est_file" "$ALIGN" --save_results "$ape_zip"
            run_cmd evo_rpe kitti "$GT_FILE" "$est_file" "$ALIGN" --save_results "$rpe_zip"
        else
            run_cmd evo_ape kitti "$GT_FILE" "$est_file" --save_results "$ape_zip"
            run_cmd evo_rpe kitti "$GT_FILE" "$est_file" --save_results "$rpe_zip"
        fi
    done

    shopt -s nullglob
    ape_files=("$config_dir"/ape_*.zip)
    rpe_files=("$config_dir"/rpe_*.zip)
    shopt -u nullglob

    if [[ "${#ape_files[@]}" -gt 0 ]]; then
        run_cmd evo_res "${ape_files[@]}" --save_table "$config_dir/tabela_resumo_APE.csv"
    else
        echo "   [AVISO] Nenhum resultado APE para $config"
    fi

    if [[ "${#rpe_files[@]}" -gt 0 ]]; then
        run_cmd evo_res "${rpe_files[@]}" --save_table "$config_dir/tabela_resumo_RPE.csv"
    else
        echo "   [AVISO] Nenhum resultado RPE para $config"
    fi
done

if [[ "$found" -eq 0 ]]; then
    echo "Nenhuma trajetoria foi encontrada. Verifique --results-dir, --tag e --configs." >&2
    exit 1
fi

echo
echo "=================================================="
echo " GERANDO TABELAS GERAIS"
echo "=================================================="

shopt -s nullglob
all_ape=("$METRICS_DIR"/*/ape_*.zip)
all_rpe=("$METRICS_DIR"/*/rpe_*.zip)
shopt -u nullglob

if [[ "${#all_ape[@]}" -gt 0 ]]; then
    run_cmd evo_res "${all_ape[@]}" --save_table "$METRICS_DIR/tabela_resumo_APE_param_sweep.csv"
fi

if [[ "${#all_rpe[@]}" -gt 0 ]]; then
    run_cmd evo_res "${all_rpe[@]}" --save_table "$METRICS_DIR/tabela_resumo_RPE_param_sweep.csv"
fi

echo
echo "SUCESSO. Metricas salvas em: $METRICS_DIR"
