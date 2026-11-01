#!/usr/bin/env bash
# Run bench/tf_agent.py on a Bench2Drive route file against CARLA 0.9.15 on Windows.
#
#   bash bench/run.sh dev10 models/transfuser.pt            # results in data/bench/dev10_transfuser/
#   bash bench/run.sh bench2drive220 models/tf_camera.pt
#   ROUTES_SUBSET=0-2 bash bench/run.sh dev10 models/transfuser.pt
#
# If the evaluator or the server dies, it restarts both and resumes at the
# route it was on. A route is retried MAX_ROUTE_TRIES times; after that the
# evaluator's checkpoint is left as it is and the run stops, so a route that
# always crashes is visible rather than silently skipped.
set -uo pipefail

SET=$1
MODEL=$(realpath "$2")
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
BENCH=${BENCH_ROOT:-$HOME/workspace/bench}
B2D=$BENCH/Bench2Drive
WIN_CARLA='C:\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe'
MAX_ROUTE_TRIES=${MAX_ROUTE_TRIES:-4}
PORT=2000

case $SET in
  dev10) ROUTES=leaderboard/data/drivetransformer_bench2drive_dev10.xml ;;
  bench2drive220) ROUTES=leaderboard/data/bench2drive220.xml ;;
  *) ROUTES=$SET ;;
esac

OUT=${BENCH_OUT:-$PROJECT/data/bench/${SET}_$(basename "$MODEL" .pt)}
mkdir -p "$OUT"
HOST=${CARLA_HOST:-$(ip route show default | awk '{print $3; exit}')}

export CARLA_EXTERNAL_SERVER=1
export CARLA_ROOT=/mnt/c/CARLA_0.9.15/WindowsNoEditor
export SAVE_PATH=$OUT
export LEADERBOARD_ROOT=$B2D/leaderboard
export SCENARIO_RUNNER_ROOT=$B2D/scenario_runner
export PYTHONPATH=$B2D/leaderboard:$B2D/scenario_runner:$PROJECT:$BENCH/carla_pythonapi
export TF_AGENT_DEBUG=${TF_AGENT_DEBUG:-}

# An open port is not enough: C:\CARLA is 0.9.16, which has the wrong maps and scenarios.
server_up() {
  timeout 3 bash -c "</dev/tcp/$HOST/$PORT" 2>/dev/null || return 1
  powershell.exe -NoProfile -Command \
    "(Get-Process CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue).Path" </dev/null | grep -q 'CARLA_0.9.15'
}

start_server() {
  taskkill.exe /F /IM CarlaUE4-Win64-Shipping.exe /T >/dev/null 2>&1
  taskkill.exe /F /IM CarlaUE4.exe /T >/dev/null 2>&1
  sleep 5
  echo "[run] starting CARLA 0.9.15"
  powershell.exe -NoProfile -Command \
    "Start-Process -FilePath '$WIN_CARLA' -ArgumentList '-quality-level=Epic','-windowed','-ResX=320','-ResY=240','-nosound','-carla-rpc-port=$PORT'" </dev/null
  for _ in $(seq 60); do
    server_up && { sleep 15; return 0; }
    sleep 5
  done
  echo "[run] server did not come up"
  return 1
}

routes_done() {
  python3 - "$OUT/results.json" <<'PY'
import json, sys
try:
    print(len(json.load(open(sys.argv[1]))["_checkpoint"]["records"]))
except Exception:
    print(0)
PY
}

cd "$B2D" || exit 1
SUBSET=()
[ -n "${ROUTES_SUBSET:-}" ] && SUBSET=(--routes-subset="$ROUTES_SUBSET")

last_done=-1
tries=0
while :; do
  server_up || start_server || exit 1
  # Unset rather than False: --resume is type=bool, so any string means True.
  "$BENCH/.venv/bin/python" leaderboard/leaderboard/leaderboard_evaluator.py \
    --host="$HOST" --port=$PORT --traffic-manager-port=8000 \
    --routes="$ROUTES" --repetitions=1 --track=SENSORS \
    --checkpoint="$OUT/results.json" --debug-checkpoint="$OUT/live.txt" \
    --agent="$PROJECT/bench/tf_agent.py" --agent-config="$MODEL" \
    --resume=True --timeout=600 "${SUBSET[@]}"
  code=$?
  [ $code -eq 0 ] && { echo "[run] finished"; exit 0; }

  done_now=$(routes_done)
  if [ "$done_now" = "$last_done" ]; then tries=$((tries + 1)); else tries=1; last_done=$done_now; fi
  echo "[run] evaluator exit $code at $done_now routes, try $tries/$MAX_ROUTE_TRIES"
  [ "$tries" -ge "$MAX_ROUTE_TRIES" ] && { echo "[run] giving up on this route"; exit 1; }
  start_server || exit 1
done
