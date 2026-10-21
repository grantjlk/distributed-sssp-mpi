#include "dijkstra.h"
#include "logger.h"
#include "timer.h"
#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <climits>

static const int INF = INT_MAX / 2;

std::vector<int> run_dijkstra(
    const Graph& g,
    const Partition& p,
    int source,
    int rank,
    int num_ranks,
    Metrics& metrics)
{
    MpiTimer timer;
    timer.start();

    int num_nodes = g.num_nodes;

    std::vector<int> dist(num_nodes, INF);
    std::vector<bool> settled(num_nodes, false);
    dist[source] = 0;

    metrics.messages_sent = 0;
    metrics.bytes_sent    = 0;
    metrics.iterations    = 0;

    // only scan the nodes this rank owns when looking for the local minimum
    std::vector<int> owned = get_owned_nodes(p, rank);

    // (distance, node id) pair for MPI_MINLOC
    struct DistNode { int dist; int node; };

    for (int iter = 0; iter < num_nodes; iter++) {
        metrics.iterations++;

        // each rank finds its local best unsettled node
        DistNode local_best = {INF, -1};
        for (int i : owned) {
            if (!settled[i] && dist[i] < local_best.dist) {
                local_best = {dist[i], i};
            }
        }

        // one MINLOC reduction gives every rank both the global min distance
        // and the node that has it (ties go to the smaller node id)
        DistNode global_best;
        // checked mpi all reduce call
        int rc = MPI_Allreduce(
            &local_best,
            &global_best,
            1,
            MPI_2INT,
            MPI_MINLOC,
            MPI_COMM_WORLD
        );
        if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);

        metrics.messages_sent += num_ranks;
        metrics.bytes_sent    += num_ranks * sizeof(DistNode);

        // no more reachable unsettled nodes
        if (global_best.dist == INF) break;

        int u = global_best.node;
        settled[u] = true;

        // all ranks relax edges of u since all have full graph
        for (const auto& [v, w] : g.adj[u]) {
            // inf guard ensures addition cant overflow before we do it
            if (!settled[v] && dist[u] < INF - w && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }

    metrics.runtime_seconds = timer.stop();

    if (rank == 0) {
        int reachable = 0;
        for (int i = 0; i < num_nodes; i++) {
            if (dist[i] < INF) reachable++;
        }
        LOG_INFO("[dijkstra] Source: " << source);
        LOG_INFO("[dijkstra] Reachable nodes: " << reachable << " / " << num_nodes);
    }

    return dist;
}