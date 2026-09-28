#!/bin/bash
# Usage: ./run/finalize.sh [--dry-run] <session_id> [<session_id> ...]
#        ./run/finalize.sh [--dry-run] --all
#
# For each discovery session that is finished (hit max_iterations or ran out of
# patience, and its last iteration has no parse failures), rebuild the Pareto
# frontier (validation pairwise accuracy vs joint_top1_rate) from the saved
# predictions and write harness/<id>/iter{NNN}_pairwise{X}_joint1{Y}.py. No model
# is called. Updates history.json and summary.json (pareto_frontier_iterations),
# which run/held-out-ranked.sh reads. Old *_locf1*.py harnesses are moved into
# harness/<id>/legacy_locf1/. Unfinished sessions are reported and left alone.
#
# The session's config is found as meta-vul/configs/.generated-config-*<id>.yaml
# (falling back to config-titanvul-cweval-generic.yaml). A session missing from
# meta-vul/outputs/ is staged from ../results/<id>/ first.
set -u
cd "$(dirname "$0")/.." || exit 1
REPO_DIR="$(pwd)"

DRY=(); SESSIONS=(); ALL=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=(--dry-run) ;;
    --all) ALL=1 ;;
    -h|--help) sed -n '2,3p' "$0"; exit 0 ;;
    *) SESSIONS+=("$a") ;;
  esac
done
if [ "$ALL" = 1 ]; then
  for h in meta-vul/outputs/*/history.json; do SESSIONS+=("$(basename "$(dirname "$h")")"); done
fi
[ ${#SESSIONS[@]} -eq 0 ] && { sed -n '2,3p' "$0" >&2; exit 1; }

find_config() {  # find_config <session_id> -> config path relative to meta-vul/
  local sid="$1" alt="${1/_seed/-seed}" c
  for pat in "$sid" "$alt"; do
    c="$(ls meta-vul/configs/.generated-config-*"$pat".yaml 2>/dev/null | head -1)"
    [ -n "$c" ] && { echo "${c#meta-vul/}"; return; }
  done
  echo "configs/config-titanvul-cweval-generic.yaml"
}

ROWS=()
for sid in "${SESSIONS[@]}"; do
  out="meta-vul/outputs/$sid"
  if [ ! -f "$out/history.json" ]; then
    backup="$REPO_DIR/../../results/$sid/meta-vul/outputs/$sid"
    if [ -f "$backup/history.json" ] && [ ${#DRY[@]} -eq 0 ]; then
      echo "[finalize] staging $sid from $backup"
      mkdir -p "$out" && rsync -a "$backup/" "$out/"
    else
      ROWS+=("$sid|no history.json|"); continue
    fi
  fi
  cfg="$(find_config "$sid")"
  echo "=== $sid  (config $cfg)"
  log="$(cd meta-vul && python3 run_optimization.py --config "$cfg" --finalize "$sid" ${DRY[@]+"${DRY[@]}"} 2>&1)"
  rc=$?
  echo "$log"
  frontier="$(printf '%s\n' "$log" | sed -n "s/^FRONTIER $sid //p")"
  case $rc in
    0) status="finalized${DRY:+ (dry run)}" ;;
    3) status="$(printf '%s\n' "$log" | sed -n 's/.*UNFINISHED (\(.*\))$/unfinished: \1/p')" ;;
    4) status="$(printf '%s\n' "$log" | sed -n 's/.*MISSING predictions for iterations \(.*\);.*/missing predictions: \1/p')" ;;
    *) status="error (exit $rc)" ;;
  esac
  ROWS+=("$sid|$status|$frontier")
done

echo
printf '%-45s  %-60s  %s\n' SESSION STATUS FRONTIER
for r in "${ROWS[@]}"; do
  IFS='|' read -r s st fr <<<"$r"
  printf '%-45s  %-60s  %s\n' "$s" "$st" "$fr"
done
