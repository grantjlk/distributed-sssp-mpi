#include <mpi.h>
#include <iostream>
#include <string>
#include <climits>
#include "graph.h"
#include "partition.h"
#include "metrics.h"
#include "leader_election.h"
#include "dijkstra.h"
#include "logger.h"


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // parse CLI arguments
    std::string graph_path, part_path, algo;
    int source = 0;
    int rounds = 200;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--graph" && i + 1 < argc)       graph_path = argv[++i];
        else if (arg == "--part" && i + 1 < argc)   part_path  = argv[++i];
        else if (arg == "--algo" && i + 1 < argc)   algo       = argv[++i];
        else if (arg == "--source" && i + 1 < argc) source     = std::stoi(argv[++i]);
        else if (arg == "--rounds" && i + 1 < argc) rounds     = std::stoi(argv[++i]);
    }

    if (graph_path.empty() || part_path.empty() || algo.empty()) {
        if (rank == 0) {
            LOG_ERROR("Usage: ngs_mpi --graph <path> --part <path> --algo <leader|dijkstra-baseline> [--source 0] [--rounds 200]");
        }
        MPI_Finalize();
        return 1;
    }

    // "dijkstra" is kept as an alias so older commands still work
    if (algo == "dijkstra") algo = "dijkstra-baseline";

    if (algo != "leader" && algo != "dijkstra-baseline") {
        if (rank == 0) {
            LOG_ERROR("Unknown algorithm '" << algo << "'");
        }
        MPI_Finalize();
        return 1;
    }

    if (rounds <= 0) {
        if (rank == 0) {
            LOG_ERROR("--rounds must be a positive integer");
        }
        MPI_Finalize();
        return 1;
    }

    // load graph and partition on all ranks
    Graph g;
    Partition p;
    int load_ok = 1;

    try {
        g = load_graph(graph_path);
        p = load_partition(part_path);
    } catch (const std::exception& ex) {
        LOG_ERROR("Error loading files: " << ex.what());
        load_ok = 0;
    }

    // all ranks agree on whether loading succeeded
    int global_ok = 0;
    MPI_Allreduce(&load_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (global_ok == 0) {
        MPI_Finalize();
        return 1;
    }

    if (p.num_ranks != size) {
        if (rank == 0) {
            LOG_ERROR("Partition mismatch: " << p.num_ranks << " ranks in file, but launched with " << size);
        }
        MPI_Finalize();
        return 1;
    }

    if (p.num_nodes != g.num_nodes) {
        if (rank == 0) {
            LOG_ERROR("Node mismatch: partition has " << p.num_nodes << " nodes, graph has " << g.num_nodes);
        }
        MPI_Finalize();
        return 1;
    }

    if (source < 0 || source >= g.num_nodes) {
        if (rank == 0) {
            LOG_ERROR("Source node " << source << " out of range [0, " << g.num_nodes - 1 << "]");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        LOG_INFO("Running algorithm: " << algo << " with " << size << " ranks");
    }

    // leader election algorithm
    Metrics metrics = {0, 0, 0, 0.0};

    if (algo == "leader") {
        int leader = run_leader_election(g, p, rank, size, rounds, metrics);
        
        // verify all ranks agree on the same leader
        std::vector<int> all_leaders(size);
        MPI_Gather(&leader, 1, MPI_INT,
                all_leaders.data(), 1, MPI_INT,
                0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            print_metrics("Leader Election", metrics);
            bool all_agree = true;
            for (int i = 0; i < size; i++) {
                if (all_leaders[i] != leader) {
                    all_agree = false;
                    LOG_ERROR("[leader] DISAGREEMENT: rank " << i << " has leader " << all_leaders[i]);
                }
            }
            if (all_agree) {
                LOG_INFO("[leader] SUCCESS: All ranks verified agreement on leader: " << leader);
            }
        }
    }
    else if (algo == "dijkstra-baseline") {
    std::vector<int> dist = run_dijkstra(g, p, source, rank, size, metrics);
    if (rank == 0) {
        print_metrics("Dijkstra", metrics);
        // print first 10 distances as sanity check
        LOG_INFO("[dijkstra] Sample distances from source " << source << ":");
        for (int i = 0; i < std::min(10, g.num_nodes); i++) {
            if (dist[i] < (INT_MAX / 2)) {
                LOG_INFO("  node " << i << ": " << dist[i]);
            } else {
                LOG_INFO("  node " << i << ": unreachable");
            }
        }
    }
}

    MPI_Finalize();
    return 0;
}