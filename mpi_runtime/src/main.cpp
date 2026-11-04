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
#include "local_graph.h"
#include "delta_stepping.h"
#include "sssp_reference.h"
#include "csr.h"
#include "timer.h"
#include <vector>

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Prints how the block partition splits the graph: per-rank node and edge
// counts, and cut edges (edges whose target lives on another rank)
static void print_partition_stats(const LocalGraph& lg, double load_seconds) {
    long long stats[3] = {lg.num_owned(), lg.num_local_edges(), 0};
    for (int32_t v : lg.targets) {
        if (!lg.owns(v)) stats[2]++;
    }

    std::vector<long long> all(3 * lg.num_ranks);
    MPI_Gather(stats, 3, MPI_LONG_LONG, all.data(), 3, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    if (lg.rank != 0) return;

    long long total_cut = 0, max_edges = 0;
    std::cout << "\n=== Partition: " << lg.num_ranks << " block(s) ===\n";
    for (int r = 0; r < lg.num_ranks; r++) {
        long long nodes = all[3 * r], edges = all[3 * r + 1], cut = all[3 * r + 2];
        total_cut += cut;
        max_edges = std::max(max_edges, edges);
        std::cout << "  Rank " << r << ": " << nodes << " nodes, " << edges
                  << " edges, " << cut << " cut\n";
    }
    double avg_edges = (double)lg.num_edges / lg.num_ranks;
    std::cout << "  Edge cut:       " << total_cut << " / " << lg.num_edges << " ("
              << 100.0 * total_cut / std::max<int64_t>(lg.num_edges, 1) << "%)\n"
              << "  Edge imbalance: " << max_edges / std::max(avg_edges, 1.0)
              << " (max rank edges / average)\n"
              << "  Load time:      " << load_seconds << " s (slowest rank)\n"
              << "========================\n\n";
}

// Gathers every rank's distances on rank 0 and compares them with a
// sequential Dijkstra over the full graph. Returns true on all ranks if they match.
static bool verify_distances(const LocalGraph& lg, const std::vector<int64_t>& dist,
                             const std::string& graph_path, int32_t source) {
    // blocks are contiguous and in rank order, so gathering by rank
    // produces the distances in global node order
    std::vector<int> counts(lg.num_ranks), displs(lg.num_ranks);
    for (int r = 0; r < lg.num_ranks; r++) {
        displs[r] = block_begin(lg.num_nodes, lg.num_ranks, r);
        counts[r] = block_begin(lg.num_nodes, lg.num_ranks, r + 1) - displs[r];
    }

    std::vector<int64_t> all;
    if (lg.rank == 0) all.resize(lg.num_nodes);
    MPI_Gatherv(dist.data(), lg.num_owned(), MPI_INT64_T,
                all.data(), counts.data(), displs.data(), MPI_INT64_T, 0, MPI_COMM_WORLD);

    int ok = 1;
    if (lg.rank == 0) {
        std::vector<int64_t> expected = sequential_sssp(read_csr(graph_path), source);
        long long mismatches = 0;
        for (int64_t v = 0; v < lg.num_nodes; v++) {
            if (all[v] != expected[v]) {
                if (mismatches < 5) {
                    LOG_ERROR("[verify] node " << v << ": got " << all[v]
                              << ", expected " << expected[v]);
                }
                mismatches++;
            }
        }
        ok = mismatches == 0;
        if (ok) {
            LOG_INFO("[verify] PASSED: all " << lg.num_nodes << " distances match sequential Dijkstra");
        } else {
            LOG_ERROR("[verify] FAILED: " << mismatches << " of " << lg.num_nodes << " distances differ");
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ok == 1;
}

struct CsrOptions {
    std::string graph_path;
    std::string algo;
    int source = 0;
    int64_t delta = 0;   // 0 = use the average edge weight
    bool verify = false;
};

static int run_delta_mode(const LocalGraph& lg, const CsrOptions& opt) {
    if (opt.source < 0 || opt.source >= lg.num_nodes) {
        if (lg.rank == 0) {
            LOG_ERROR("Source node " << opt.source << " out of range [0, " << lg.num_nodes - 1 << "]");
        }
        return 1;
    }

    int64_t delta = opt.delta;
    if (delta <= 0) {
        long long local_sum = 0, global_sum = 0;
        for (int32_t w : lg.weights) local_sum += w;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        delta = std::max<int64_t>(1, global_sum / std::max<int64_t>(lg.num_edges, 1));
    }

    DeltaStats stats;
    std::vector<int64_t> dist = run_delta_stepping(lg, opt.source, delta, stats);

    long long local_reached = 0, global_reached = 0;
    for (int64_t d : dist) {
        if (d < SSSP_INF) local_reached++;
    }
    MPI_Reduce(&local_reached, &global_reached, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (lg.rank == 0) {
        LOG_INFO("[delta] Source: " << opt.source);
        LOG_INFO("[delta] Reachable nodes: " << global_reached << " / " << lg.num_nodes);
        print_delta_stats(stats, delta);
    }

    if (opt.verify && !verify_distances(lg, dist, opt.graph_path, opt.source)) {
        return 1;
    }
    return 0;
}

// Distributed mode: each rank loads only its block of a .csr graph
static int run_csr_mode(const CsrOptions& opt, int rank, int size) {
    const std::string& graph_path = opt.graph_path;
    const std::string& algo = opt.algo;
    if (algo != "partition-stats" && algo != "delta") {
        if (rank == 0) {
            LOG_ERROR("Unknown algorithm '" << algo << "' for .csr graphs (available: delta, partition-stats)");
        }
        return 1;
    }

    LocalGraph lg;
    int load_ok = 1;
    MpiTimer timer;
    timer.start();
    try {
        lg = load_graph_local(graph_path, rank, size);
    } catch (const std::exception& ex) {
        LOG_ERROR("Error loading graph: " << ex.what());
        load_ok = 0;
    }
    double load_seconds = timer.stop();

    int global_ok = 0;
    MPI_Allreduce(&load_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (global_ok == 0) return 1;

    if (rank == 0) {
        LOG_INFO("[graph] " << graph_path << ": " << lg.num_nodes << " nodes, "
                 << lg.num_edges << " edges across " << size << " ranks");
    }

    if (algo == "delta") {
        return run_delta_mode(lg, opt);
    }

    print_partition_stats(lg, load_seconds);
    return 0;
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // parse CLI arguments
    std::string graph_path, part_path, algo;
    int source = 0;
    int rounds = 200;
    int64_t delta = 0;
    bool verify = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--graph" && i + 1 < argc)       graph_path = argv[++i];
        else if (arg == "--part" && i + 1 < argc)   part_path  = argv[++i];
        else if (arg == "--algo" && i + 1 < argc)   algo       = argv[++i];
        else if (arg == "--source" && i + 1 < argc) source     = std::stoi(argv[++i]);
        else if (arg == "--rounds" && i + 1 < argc) rounds     = std::stoi(argv[++i]);
        else if (arg == "--delta" && i + 1 < argc)  delta      = std::stoll(argv[++i]);
        else if (arg == "--verify")                 verify     = true;
    }

    // .csr graphs are block-partitioned in-process and need no --part file
    if (ends_with(graph_path, ".csr") && !algo.empty()) {
        CsrOptions opt;
        opt.graph_path = graph_path;
        opt.algo       = algo;
        opt.source     = source;
        opt.delta      = delta;
        opt.verify     = verify;
        int rc = run_csr_mode(opt, rank, size);
        MPI_Finalize();
        return rc;
    }

    if (graph_path.empty() || part_path.empty() || algo.empty()) {
        if (rank == 0) {
            LOG_ERROR("Usage: ngs_mpi --graph <path> --part <path> --algo <leader|dijkstra-baseline> [--source 0] [--rounds 200]");
            LOG_ERROR("       ngs_mpi --graph <file.csr> --algo <delta|partition-stats> [--source 0] [--delta D] [--verify]");
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