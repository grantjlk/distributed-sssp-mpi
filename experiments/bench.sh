#!/bin/bash
# Times sequential Dijkstra and delta-stepping on one .csr graph and appends
# one CSV row per run to experiments/results/<name>.csv.
#
#   ./experiments/bench.sh <graph.csr> <name> "<deltas>" "<ranks>"
#   ./experiments/bench.sh rmat20.csr rmat20 "1" "1 2 4 6"                    # scaling
#   RUNS=5 ./experiments/bench.sh rmat20.csr sweep_rmat20 "1 2 5 10 20" "1 6"  # delta sweep
#
# RUNS (default 10) sources are drawn from a fixed seed, keeping only ones
# that reach at least half the graph, so every config runs the same sources.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GRAPH=$1; NAME=$2; DELTAS=$3; RANKS=$4
RUNS=${RUNS:-10}
if [ -z "$RANKS" ]; then
    echo "usage: $0 <graph.csr> <name> \"<deltas>\" \"<ranks>\"" >&2
    exit 1
fi

BIN=./build/sssp_mpi
OUT=experiments/results/$NAME.csv
mkdir -p experiments/results
[ -f "$OUT" ] || echo "graph,algo,ranks,delta,source,seconds,reached,buckets,phases,relaxations,remote_requests,bytes_sent" > "$OUT"

# pull the numbers out of the program's printed metrics
parse() {
    awk -F': *' '
        /Reachable nodes/ { split($NF, a, " "); reached = a[1]; n = a[3] }
        /Runtime:/        { secs = $2 + 0 }
        /Buckets:/        { b = $2 }
        /Light phases:/   { p = $2 }
        /Relaxations:/    { x = $2 }
        /Remote requests:/{ q = $2 }
        /Bytes sent:/     { y = $2 }
        END {
            if (secs == "") { print "no runtime in output" > "/dev/stderr"; exit 1 }
            printf "%s,%s,%s,%s,%s,%s,%s,%s\n", secs, reached, b, p, x, q, y, n
        }'
}

# pick sources and time the sequential baseline at the same time
RANDOM=42
sources=()
tries=0
n=1
while [ ${#sources[@]} -lt "$RUNS" ]; do
    tries=$((tries + 1))
    [ $tries -gt $((RUNS * 50)) ] && { echo "could not find $RUNS good sources" >&2; exit 1; }
    src=$(( ((RANDOM << 15) | RANDOM) % n ))
    row=$(mpirun -n 1 $BIN --graph "$GRAPH" --algo seq --source $src | parse)
    n=${row##*,}
    reached=$(echo "$row" | cut -d, -f2)
    if [ $tries -eq 1 ]; then continue; fi    # first call only learns n
    if [ $((reached * 2)) -ge "$n" ]; then
        sources+=($src)
        echo "$NAME,seq,1,,$src,${row%,*}" >> "$OUT"
    fi
done
echo "[$NAME] sources: ${sources[*]}"

for delta in $DELTAS; do
    for r in $RANKS; do
        for src in "${sources[@]}"; do
            row=$(mpirun -n $r --bind-to core $BIN --graph "$GRAPH" --algo delta \
                  --delta $delta --source $src | parse)
            echo "$NAME,delta,$r,$delta,$src,${row%,*}" >> "$OUT"
        done
        echo "[$NAME] delta=$delta ranks=$r done"
    done
done
