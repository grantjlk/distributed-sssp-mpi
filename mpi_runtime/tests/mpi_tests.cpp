#include <gtest/gtest.h>
#include <mpi.h>
#include <climits>
#include <algorithm>
#include "graph.h"
#include "partition.h"
#include "leader_election.h"
#include "dijkstra.h"
#include "metrics.h"
#include "exchange.h"
#include "csr.h"
#include "local_graph.h"
#include "delta_stepping.h"
#include "sssp_reference.h"
#include <cstdio>

int rank, size;

TEST(LeaderElectionTest, CorrectLeaderElected) {
    Graph g = load_graph("tests/small_graph.json");
    Partition p = load_partition("tests/small_part.json");
    if (p.num_ranks != size) GTEST_SKIP() << "v1 partition file is for " << p.num_ranks << " ranks";
    Metrics m = {0, 0, 0, 0.0};

    int leader = run_leader_election(g, p, rank, size, 100, m);

    // leader must be node 4 (highest id in 0-4)
    EXPECT_EQ(leader, 4);
}

TEST(LeaderElectionTest, AllRanksAgree) {
    Graph g = load_graph("tests/small_graph.json");
    Partition p = load_partition("tests/small_part.json");
    if (p.num_ranks != size) GTEST_SKIP() << "v1 partition file is for " << p.num_ranks << " ranks";
    Metrics m = {0, 0, 0, 0.0};

    int leader = run_leader_election(g, p, rank, size, 100, m);

    // gather all leaders to rank 0 and check they all agree
    std::vector<int> all_leaders(size);
    MPI_Gather(&leader, 1, MPI_INT,
               all_leaders.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            EXPECT_EQ(all_leaders[i], leader)
                << "Rank " << i << " disagrees on leader";
        }
    }
}

TEST(DijkstraTest, CorrectDistances) {
    Graph g = load_graph("tests/small_graph.json");
    Partition p = load_partition("tests/small_part.json");
    if (p.num_ranks != size) GTEST_SKIP() << "v1 partition file is for " << p.num_ranks << " ranks";
    Metrics m = {0, 0, 0, 0.0};

    std::vector<int> dist = run_dijkstra(g, p, 0, rank, size, m);

    if (rank == 0) {
        EXPECT_EQ(dist[0], 0);   // source
        EXPECT_EQ(dist[1], 8);   // 0->1
        EXPECT_EQ(dist[2], 11);  // 0->3->4->2 = 7+2+2
        EXPECT_EQ(dist[3], 7);   // 0->3
        EXPECT_EQ(dist[4], 9);   // 0->3->4
    }
}

TEST(DijkstraTest, AllNodesReachable) {
    Graph g = load_graph("tests/small_graph.json");
    Partition p = load_partition("tests/small_part.json");
    if (p.num_ranks != size) GTEST_SKIP() << "v1 partition file is for " << p.num_ranks << " ranks";
    Metrics m = {0, 0, 0, 0.0};

    std::vector<int> dist = run_dijkstra(g, p, 0, rank, size, m);

    if (rank == 0) {
        for (int i = 0; i < g.num_nodes; i++) {
            EXPECT_LT(dist[i], INT_MAX / 2)
                << "Node " << i << " is unreachable";
        }
    }
}

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
            if (size > 1) EXPECT_GT(stats.remote_requests, 0);
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