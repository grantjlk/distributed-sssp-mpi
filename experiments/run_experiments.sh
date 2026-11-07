#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GRAPH=outputs/graph.json
RESULTS=experiments/results.txt
> "$RESULTS"

echo "=== Experiment 1: Partitioning Strategy Comparison (4 ranks) ===" | tee -a "$RESULTS"
for strategy in contiguous random; do
    python3 tools/partition/partition.py "$GRAPH" --ranks 4 \
        --out outputs/part_exp1_${strategy}.json --strategy $strategy >> "$RESULTS"

    echo "--- Leader Election [$strategy] ---" | tee -a "$RESULTS"
    mpirun --oversubscribe -n 4 ./build/ngs_mpi --graph "$GRAPH" \
        --part outputs/part_exp1_${strategy}.json \
        --algo leader --rounds 200 2>&1 | tee -a "$RESULTS"

    echo "--- Dijkstra [$strategy] ---" | tee -a "$RESULTS"
    mpirun --oversubscribe -n 4 ./build/ngs_mpi --graph "$GRAPH" \
        --part outputs/part_exp1_${strategy}.json \
        --algo dijkstra-baseline --source 0 2>&1 | tee -a "$RESULTS"
done

echo "" | tee -a "$RESULTS"
echo "=== Experiment 2: Scalability (contiguous, vary ranks) ===" | tee -a "$RESULTS"
for ranks in 1 2 4 8; do
    python3 tools/partition/partition.py "$GRAPH" --ranks $ranks \
        --out outputs/part_exp2_${ranks}.json --strategy contiguous >> "$RESULTS"

    echo "--- Leader Election [ranks=$ranks] ---" | tee -a "$RESULTS"
    mpirun --oversubscribe -n $ranks ./build/ngs_mpi --graph "$GRAPH" \
        --part outputs/part_exp2_${ranks}.json \
        --algo leader --rounds 200 2>&1 | tee -a "$RESULTS"

    echo "--- Dijkstra [ranks=$ranks] ---" | tee -a "$RESULTS"
    mpirun --oversubscribe -n $ranks ./build/ngs_mpi --graph "$GRAPH" \
        --part outputs/part_exp2_${ranks}.json \
        --algo dijkstra-baseline --source 0 2>&1 | tee -a "$RESULTS"
done

echo "" | tee -a "$RESULTS"
echo "=== Done. Results saved to $RESULTS ==="