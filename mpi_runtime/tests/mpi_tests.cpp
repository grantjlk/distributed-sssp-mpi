#include <gtest/gtest.h>
#include <mpi.h>
#include <climits>
#include "graph.h"
#include "partition.h"
#include "leader_election.h"
#include "dijkstra.h"
#include "metrics.h"
#include "exchange.h"

int rank, size;

TEST(LeaderElectionTest, CorrectLeaderElected) {
    Graph g = load_graph("tests/small_graph.json");
    Partition p = load_partition("tests/small_part.json");
    Metrics m = {0, 0, 0, 0.0};

    int leader = run_leader_election(g, p, rank, size, 100, m);

    // leader must be node 4 (highest id in 0-4)
    EXPECT_EQ(leader, 4);
}

TEST(LeaderElectionTest, AllRanksAgree) {
    Graph g = load_graph("tests/small_graph.json");
    Partition p = load_partition("tests/small_part.json");
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}