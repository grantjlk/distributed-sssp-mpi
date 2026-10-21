#include "leader_election.h"
#include "logger.h"
#include <mpi.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>

int run_leader_election(
    const Graph& g,
    const Partition& p,
    int rank,
    int num_ranks,
    int rounds,
    Metrics& metrics)
{
    auto start = std::chrono::high_resolution_clock::now();

    int num_nodes = g.num_nodes;

    // candidate[i] = best leader candidate node i has seen so far
    // initialize each node with its own id
    std::vector<int> candidate(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        candidate[i] = i;
    }

    metrics.messages_sent = 0;
    metrics.bytes_sent    = 0;
    metrics.iterations    = 0;

    std::vector<int> owned = get_owned_nodes(p, rank);

    for (int round = 0; round < rounds; round++) {
        metrics.iterations++;
        bool changed = false;

        // --- local update ---
        // for each owned node, check all neighbors and update candidate
        for (int i : owned) {
            for (const auto& [neighbor, weight] : g.adj[i]) {
                if (candidate[neighbor] > candidate[i]) {
                    candidate[i] = candidate[neighbor];
                    changed = true;
                }
            }
        }

        // --- share candidates across all ranks ---
        // All ranks hold the full candidate[] vector, so candidate[neighbor]
        // always reflects the post-Allreduce value from the previous round.
        // MPI_Allreduce(MAX) merges local updates globally — no propagation
        // is ever missed since every rank sees all edges during local update.
        std::vector<int> global_candidate(num_nodes);
        // checked mpi call
        int rc1 = MPI_Allreduce(
            candidate.data(),
            global_candidate.data(),
            num_nodes,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD
        );
        if (rc1 != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc1);

        metrics.messages_sent += num_ranks;
        metrics.bytes_sent    += num_ranks * num_nodes * sizeof(int);

        // track if anything changed globally
        int local_changed  = changed ? 1 : 0;
        int global_changed = 0;
        // checked mpi call
        int rc2 = MPI_Allreduce(
            &local_changed,
            &global_changed,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD
        );
        if (rc2 != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc2);

        metrics.messages_sent += num_ranks;
        metrics.bytes_sent    += num_ranks * sizeof(int);
        

        candidate = global_candidate;

        // early termination if nothing changed
        if (global_changed == 0) {
            if (rank == 0) {
                LOG_INFO("[leader] Converged early at round " << round);
            }
            break;
        }
    }

    // all nodes should agree on same leader = max node id seen
    int leader = *std::max_element(candidate.begin(), candidate.end());

    auto end = std::chrono::high_resolution_clock::now();
    metrics.runtime_seconds =
        std::chrono::duration<double>(end - start).count();

    if (rank == 0) {
        LOG_INFO("[leader] Elected leader: " << leader);
    }

    return leader;
}