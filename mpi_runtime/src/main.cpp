#include <mpi.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "csr.h"
#include "delta_stepping.h"
#include "local_graph.h"
#include "logger.h"
#include "sssp_reference.h"
#include "timer.h"

// whole-string integer parse: "12" is fine, "12x" or "" throws
static long long parse_int(const std::string& s) {
    size_t used = 0;
    long long v = std::stoll(s, &used);
    if (used != s.size()) throw std::invalid_argument("trailing characters");
    return v;
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
    std::vector<int64_t> expected;
    if (lg.rank == 0) {
        // a failure here must still reach the Bcast below, or the other
        // ranks would wait there forever
        try {
            expected = sequential_sssp(read_csr(graph_path), source);
        } catch (const std::exception& ex) {
            LOG_ERROR("[verify] could not build reference: " << ex.what());
            ok = 0;
        }
    }
    if (lg.rank == 0 && ok) {
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

struct Options {
    std::string graph_path;
    std::string algo;
    int source = 0;
    int64_t delta = 0;   // 0 = use the average edge weight
    bool verify = false;
};

static int run_delta_mode(const LocalGraph& lg, const Options& opt) {
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

// Sequential baseline: one process, full graph, heap Dijkstra.
// Speedups are reported against this, not against delta-stepping on 1 rank.
static int run_seq_mode(const Options& opt, int rank, int size) {
    if (size != 1) {
        if (rank == 0) {
            LOG_ERROR("--algo seq is the single-process baseline; run it with 1 rank");
        }
        return 1;
    }

    CsrGraph g;
    try {
        g = read_csr(opt.graph_path);
    } catch (const std::exception& ex) {
        LOG_ERROR("Error loading graph: " << ex.what());
        return 1;
    }
    if (opt.source < 0 || opt.source >= g.num_nodes) {
        LOG_ERROR("Source node " << opt.source << " out of range [0, " << g.num_nodes - 1 << "]");
        return 1;
    }

    double start = MPI_Wtime();
    std::vector<int64_t> dist = sequential_sssp(g, opt.source);
    double runtime = MPI_Wtime() - start;

    long long reached = 0;
    for (int64_t d : dist) {
        if (d < SSSP_INF) reached++;
    }

    LOG_INFO("[seq] Source: " << opt.source);
    LOG_INFO("[seq] Reachable nodes: " << reached << " / " << g.num_nodes);
    std::cout << "\n=== Metrics: Sequential Dijkstra ===\n"
              << std::fixed << std::setprecision(4)
              << "  Runtime:          " << runtime << " seconds\n"
              << "========================\n\n";
    return 0;
}

// Each rank loads only its block of the graph, then runs the chosen algorithm
static int run(const Options& opt, int rank, int size) {
    const std::string& graph_path = opt.graph_path;
    const std::string& algo = opt.algo;
    if (algo == "seq") {
        return run_seq_mode(opt, rank, size);
    }
    if (algo != "partition-stats" && algo != "delta") {
        if (rank == 0) {
            LOG_ERROR("Unknown algorithm '" << algo << "' (available: delta, seq, partition-stats)");
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

    // unknown flags, missing values and non-numbers are errors, so a typo
    // can't silently fall back to a default (every rank parses the same
    // arguments, so all ranks exit together)
    Options opt;
    std::string bad_arg;
    for (int i = 1; i < argc && bad_arg.empty(); i++) {
        std::string arg = argv[i];
        if (arg == "--verify") {
            opt.verify = true;
            continue;
        }
        if (i + 1 >= argc) {
            bad_arg = arg;
            break;
        }
        std::string value = argv[++i];
        try {
            if (arg == "--graph")       opt.graph_path = value;
            else if (arg == "--algo")   opt.algo       = value;
            else if (arg == "--source") opt.source     = (int)parse_int(value);
            else if (arg == "--delta")  opt.delta      = parse_int(value);
            else                        bad_arg        = arg;
        } catch (const std::exception&) {
            bad_arg = arg + " " + value;
        }
    }

    int rc = 1;
    if (!bad_arg.empty()) {
        if (rank == 0) LOG_ERROR("Invalid or unknown argument: " << bad_arg);
    } else if (opt.graph_path.empty() || opt.algo.empty()) {
        if (rank == 0) {
            LOG_ERROR("Usage: ngs_mpi --graph <file.csr> --algo <delta|seq|partition-stats> "
                      "[--source 0] [--delta D] [--verify]");
        }
    } else {
        rc = run(opt, rank, size);
    }

    MPI_Finalize();
    return rc;
}
