#pragma once
#include "local_graph.h"
#include <cstdint>
#include <vector>

// Totals across all ranks (identical on every rank after the run)
struct DeltaStats {
    long long buckets = 0;          // buckets processed (outer rounds)
    long long phases = 0;           // light-edge phases (one exchange each)
    long long relaxations = 0;      // edge relaxations attempted
    long long remote_requests = 0;  // relaxations sent to another rank
    long long bytes_sent = 0;       // bytes sent to other ranks
    double runtime_seconds = 0.0;   // slowest rank
};

// Distributed delta-stepping SSSP (Meyer & Sanders). Each rank only touches
// the nodes it owns; updates for remote nodes are batched per destination
// rank and delivered with MPI_Alltoallv once per phase.
//
// Returns distances for this rank's owned nodes (index = local id), with
// SSSP_INF for unreachable nodes. Collective over MPI_COMM_WORLD.
std::vector<int64_t> run_delta_stepping(const LocalGraph& lg, int32_t source,
                                        int64_t delta, DeltaStats& stats);

void print_delta_stats(const DeltaStats& s, int64_t delta);
