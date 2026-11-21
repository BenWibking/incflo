#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
best_dss_profile.sh rank CAMPAIGN
best_dss_profile.sh figure CAMPAIGN NEW_OUTPUT_DIR
best_dss_profile.sh convergence CAMPAIGN NEW_OUTPUT_DIR [HOLDOUT_JSON]
best_dss_profile.sh plot CAMPAIGN NEW_OUTPUT_DIR
best_dss_profile.sh profile CAMPAIGN LOCAL_INCFLO NEW_OUTPUT_DIR [--grid N] [--steps N]

CAMPAIGN is a local directory or an SSH source such as setonix:/scratch/project/run.
Optimas ranks by the campaign's Richardson or fine score. Independent searches rank by their velocity objective.
PYTHON selects the local interpreter for replay, plotfile analysis and plotting.
EOF
}
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    rank) [[ $# == 2 ]] || { usage >&2; exit 2; } ;;
    figure|plot) [[ $# == 3 ]] || { usage >&2; exit 2; } ;;
    convergence) [[ $# == 3 || $# == 4 ]] || { usage >&2; exit 2; } ;;
    profile) [[ $# -ge 4 ]] || { usage >&2; exit 2; } ;;
    *) usage >&2; exit 2 ;;
esac
action=$1
source=${2%/}/
shift 2
script_dir=$(cd "$(dirname "$0")" && pwd)
python=${PYTHON:-python3}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/dss-profile.XXXXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
mkdir "$tmp/metadata"
rsync -rtm --include='*/' --include='/config.json' --include='search.json' --include='*-score.json' \
    --exclude='*' -- "$source" "$tmp/metadata/"

"$python" - "$tmp/metadata" > "$tmp/rank.tsv" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
fine = sorted(root.rglob("fine-score.json"))
guides = sorted(root.rglob("richardson-score.json"))
config = root / "config.json"
settings = json.loads(config.read_text()) if config.is_file() else {}
richardson = bool(guides) or settings.get("mode") == "richardson"
optimas = richardson or bool(fine) or "fine_grid" in settings
rows = []
for filename in (guides if richardson else fine if optimas else root.rglob("search.json")):
    try:
        data = json.loads(filename.read_text())
        if optimas:
            value, ident = data["score"], data["trial"]
            report = filename.parent / "coarse" / "search.json"
            reference = filename.with_name("fine-score.json")
        else:
            value, ident = data.get("final_guided_return", data["final_return"]), data["random_seed"]
            report = filename
            reference = filename
        if not (isinstance(value, (int,float)) and math.isfinite(value) and value >= 0 and report.is_file() and reference.is_file()):
            raise ValueError("invalid score or missing search report")
        rows.append((value, ident, report.relative_to(root), reference.relative_to(root)))
    except (OSError, ValueError, KeyError) as error:
        print(f"Skipping {filename.relative_to(root)}: {error}", file=sys.stderr)
if not rows:
    sys.exit("No completed three-grid scores" if richardson else "No completed fine evaluations" if optimas else "No completed searches")
for value, ident, report, reference in sorted(rows):
    print(f"{value:.17g}\t{ident}\t{report}\t{reference}")
PY

if [[ $action == rank ]]; then
    cat "$tmp/rank.tsv"
    exit 0
fi
IFS=$'\t' read -r score identifier report reference < "$tmp/rank.tsv"
guide="$tmp/metadata/$(dirname "$reference")/richardson-score.json"
if [[ $action == convergence ]]; then
    [[ -f $guide ]] || { echo 'A completed Richardson trial is required for a convergence plot.' >&2; exit 1; }
    mkdir "$tmp/figure"
    if [[ $# == 2 ]]; then
        rsync -t -- "$2" "$tmp/holdout.json"
        "$python" "$script_dir/plot_dss_convergence.py" "$(dirname "$guide")" "$tmp/figure" --holdout "$tmp/holdout.json"
    else
        "$python" "$script_dir/plot_dss_convergence.py" "$(dirname "$guide")" "$tmp/figure"
    fi
    mkdir -p "$(dirname "$1")"
    mkdir "$1"
    mv "$tmp/figure/"* "$1/"
    printf 'Trial %s. Figure: %s/convergence.png\n' "$identifier" "$1"
    exit 0
fi
if [[ $action == figure ]]; then
    [[ $reference != "$report" ]] || { echo 'Use plot to render a legacy search from its endpoints.' >&2; exit 1; }
    winner="$(dirname "$reference")/fine"
    for file in vorticity-profile.png vorticity-profile.pdf radial-vorticity.csv; do
        printf '%s\n' "$winner/$file"
    done > "$tmp/files"
    mkdir "$tmp/products"
    rsync -rt --files-from="$tmp/files" -- "$source" "$tmp/products/" || {
        echo 'The winning fine profile is written at job completion. Use plot to render from existing endpoints.' >&2
        exit 1
    }
    mkdir -p "$(dirname "$1")"
    mkdir "$1"
    cp "$tmp/metadata/$reference" "$1/selected-fine.json"
    if [[ -f $guide ]]; then cp "$guide" "$1/selected-richardson.json"; fi
    mv "$tmp/products/$winner/"* "$1/"
    printf 'Trial %s, score %s. Figure: %s/vorticity-profile.png\n' "$identifier" "$score" "$1"
    exit 0
fi
native="$(dirname "$report")/candidate"
if [[ $action == profile ]]; then
    exe=$1
    out=$2
    shift 2
    [[ -x "$exe" ]] || { echo "Missing executable: $exe" >&2; exit 1; }
    printf '%s\n' "$native/velocity.bin" > "$tmp/files"
else
    out=$1
    shift
    if [[ $reference != "$report" ]]; then native="$(dirname "$reference")/fine"; fi
    read -r grid steps < <("$python" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["grids"][-1] if "grids" in r else 0, r.get("steps_by_grid",[r["steps"]])[-1])' "$tmp/metadata/$reference")
    if [[ $reference == "$report" && $grid != 0 ]]; then native="$native/n$grid"; fi
    printf -v last 'plt%05d' "$steps"
    printf '%s\n' "$native/dss_result.txt" "$native/plt00000/" "$native/$last/" > "$tmp/files"
fi
mkdir -p "$(dirname "$out")"
mkdir "$out"
mkdir "$tmp/input"
rsync -rtz --files-from="$tmp/files" -- "$source" "$tmp/input/"
cp "$tmp/metadata/$report" "$out/selected-search.json"
if [[ $reference != "$report" ]]; then
    cp "$tmp/metadata/$reference" "$out/selected-fine.json"
fi
if [[ -f $guide ]]; then cp "$guide" "$out/selected-richardson.json"; fi
printf 'Selected %s (score %s, id %s)\n' "$report" "$score" "$identifier"
if [[ $action == profile ]]; then
    mv "$tmp/input/$native/velocity.bin" "$out/accepted-velocity.bin"
    "$python" "$script_dir/dss_vorticity.py" "$out" --exe "$exe" "$@"
else
    mv "$tmp/input/$native/dss_result.txt" "$tmp/input/$native/plt00000" "$tmp/input/$native/$last" "$out/"
    "$python" "$script_dir/dss_vorticity.py" "$out" --from-plots
fi
printf 'Figure: %s/vorticity-profile.png\n' "$out"
