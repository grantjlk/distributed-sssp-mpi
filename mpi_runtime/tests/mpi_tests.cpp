#include <gtest/gtest.h>
#include <mpi.h>
#include <algorithm>
#include "exchange.h"
#include "csr.h"
#include "local_graph.h"
#include "delta_stepping.h"
#include "sssp_reference.h"
#include <cstdio>

int rank, size;

TEST(ExchangeTest, DeliversDifferentSizesToEachRank) {
    // rank s sends (s + 1) copies of 100*s + r to each rank r
    std::vector<std::vector<int>> outbox(size);
    for (int r = 0; r < size; r++) {
        outbox[r].assign(rank + 1, 100 * rank + r);
    }

    long long bytes = 0;
    std::vector<int> inbox = exchange(outbox, MPI_COMM_WORLD, bytes);

    // expect (s + 1) copies of 100*s + rank from each source s, in rank order
    std::vector<int> expected;
    for (int s = 0; s < size; s++) {
        expected.insert(expected.end(), s + 1, 100 * s + rank);
    }
    EXPECT_EQ(inbox, expected);

    // only the copies sent to other ranks count as traffic
    EXPECT_EQ(bytes, (long long)(rank + 1) * (size - 1) * (long long)sizeof(int));
}

TEST(ExchangeTest, HandlesEmptyOutboxes) {
    std::vector<std::vector<int>> outbox(size);
    long long bytes = 0;
    std::vector<int> inbox = exchange(outbox, MPI_COMM_WORLD, bytes);
    EXPECT_TRUE(inbox.empty());
    EXPECT_EQ(bytes, 0);
}

// Deterministic random undirected graph with weights 1..20. With about as
// many edges as nodes, ~13% of nodes are isolated, so unreachable nodes are
// covered too. Duplicate edges are kept, which SSSP must also handle.
static CsrGraph random_test_graph(int32_t n, int num_edges, uint32_t seed) {
    auto next = [&seed]() { seed = seed * 1664525u + 1013904223u; return seed >> 8; };

    std::vector<std::vector<std::pair<int32_t, int32_t>>> adj(n);
    for (int i = 0; i < num_edges; i++) {
        int32_t u = next() % n, v = next() % n, w = 1 + next() % 20;
        if (u == v) continue;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    CsrGraph g;
    g.num_nodes = n;
    g.offsets.push_back(0);
    for (const auto& row : adj) {
        for (auto [v, w] : row) {
            g.targets.push_back(v);
            g.weights.push_back(w);
        }
        g.offsets.push_back((int64_t)g.targets.size());
    }
    g.num_edges = (int64_t)g.targets.size();
    return g;
}

TEST(DeltaSteppingTest, MatchesSequentialDijkstra) {
    const char* path = "delta_test_graph.csr";
    CsrGraph g = random_test_graph(300, 300, 7);
    if (rank == 0) write_csr(path, g);
    MPI_Barrier(MPI_COMM_WORLD);
    LocalGraph lg = load_graph_local(path, rank, size);

    // where each rank's block goes when gathering distances in global order
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; r++) {
        displs[r] = block_begin(g.num_nodes, size, r);
        counts[r] = block_begin(g.num_nodes, size, r + 1) - displs[r];
    }

    // sources on the first and last rank; delta 1 is Dijkstra-like,
    // delta 1000 puts everything in one bucket (Bellman-Ford-like)
    for (int32_t source : {0, 299}) {
        std::vector<int64_t> expected = sequential_sssp(g, source);
        ASSERT_GT(std::count(expected.begin(), expected.end(), SSSP_INF), 0)
            << "test graph should contain unreachable nodes";

        for (int64_t delta : {1, 5, 1000}) {
            DeltaStats stats;
            std::vector<int64_t> local = run_delta_stepping(lg, source, delta, stats);

            std::vector<int64_t> all(g.num_nodes);
            MPI_Allgatherv(local.data(), lg.num_owned(), MPI_INT64_T,
                           all.data(), counts.data(), displs.data(), MPI_INT64_T, MPI_COMM_WORLD);
            EXPECT_EQ(all, expected) << "source=" << source << " delta=" << delta;

            // with several ranks, updates must actually cross rank boundaries
            if (size > 1) {
                EXPECT_GT(stats.remote_requests, 0);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::remove(path);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}