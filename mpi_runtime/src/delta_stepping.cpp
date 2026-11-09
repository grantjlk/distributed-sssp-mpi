#include "delta_stepping.h"
#include "exchange.h"
#include "sssp_reference.h"
#include "timer.h"
#include <mpi.h>
#include <climits>
#include <iomanip>
#include <iostream>

// "node v may have distance dist" sent to v's owner
struct Relaxation {
    int64_t dist;
    int32_t node;   // global id
    int32_t pad;    // keeps the struct a fixed 16 bytes
};

std::vector<int64_t> run_delta_stepping(const LocalGraph& lg, int32_t source,
                                        int64_t delta, DeltaStats& stats)
{
    stats = DeltaStats{};
    MpiTimer timer;
    timer.start();

    const int32_t n_local = lg.num_owned();
    std::vector<int64_t> dist(n_local, SSSP_INF);

    // buckets[b] holds local ids with dist in [b*delta, (b+1)*delta).
    // Entries go stale when a node later moves to a lower bucket; they are
    // skipped when their bucket is processed instead of being removed.
    std::vector<std::vector<int32_t>> buckets;

    std::vector<std::vector<Relaxation>> outbox(lg.num_ranks);
    std::vector<char> in_frontier(n_local, 0);
    std::vector<char> in_settled(n_local, 0);

    // sent_slot[v] = index of v's pending request in its owner's outbox, or -1.
    // Indexed by global id (4 bytes per node in the whole graph).
    std::vector<int32_t> sent_slot(lg.num_nodes, -1);

    // lower dist[local] if d is better and file it in the matching bucket
    auto relax_local = [&](int32_t local, int64_t d) {
        if (d < dist[local]) {
            dist[local] = d;
            size_t b = (size_t)(d / delta);
            if (b >= buckets.size()) buckets.resize(b + 1);
            buckets[b].push_back(local);
        }
    };

    // Relax the light (w <= delta) or heavy (w > delta) edges of `nodes`.
    // Targets we own are updated immediately; remote targets are batched
    // into their owner's outbox, delivered by one exchange, and applied there.
    auto relax_edges = [&](const std::vector<int32_t>& nodes, bool light) {
        for (int32_t u : nodes) {
            int64_t du = dist[u];
            for (int64_t e = lg.offsets[u]; e < lg.offsets[u + 1]; e++) {
                int32_t w = lg.weights[e];
                if ((w <= delta) != light) continue;
                int32_t v = lg.targets[e];
                stats.relaxations++;
                if (lg.owns(v)) {
                    relax_local(lg.to_local(v), du + w);
                    continue;
                }

                // send at most one request per remote node per phase,
                // keeping only the smallest candidate distance
                auto& box = outbox[lg.owner(v)];
                int32_t& slot = sent_slot[v];
                if (slot < 0) {
                    slot = (int32_t)box.size();
                    box.push_back({du + w, v, 0});
                } else if (du + w < box[slot].dist) {
                    box[slot].dist = du + w;
                }
            }
        }

        // reset only the slots used this phase, so the cost tracks
        // requests sent rather than graph size
        for (const auto& box : outbox) {
            stats.remote_requests += (long long)box.size();
            for (const Relaxation& req : box) sent_slot[req.node] = -1;
        }

        std::vector<Relaxation> inbox = exchange(outbox, MPI_COMM_WORLD, stats.bytes_sent);
        for (auto& box : outbox) box.clear();

        for (const Relaxation& req : inbox) {
            relax_local(lg.to_local(req.node), req.dist);
        }
    };

    if (lg.owns(source)) {
        relax_local(lg.to_local(source), 0);
    }

    std::vector<int32_t> frontier, settled;
    size_t k = 0;

    while (true) {
        // every rank proposes its smallest non-empty bucket, and all agree on the min
        while (k < buckets.size() && buckets[k].empty()) k++;
        long long local_k  = k < buckets.size() ? (long long)k : LLONG_MAX;
        long long global_k = LLONG_MAX;
        MPI_Allreduce(&local_k, &global_k, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
        if (global_k == LLONG_MAX) break;

        k = (size_t)global_k;
        if (k >= buckets.size()) buckets.resize(k + 1);
        stats.buckets++;
        settled.clear();

        // light phases: repeat until bucket k is empty on every rank, since
        // a light edge can drop a node back into the bucket being processed
        while (true) {
            frontier.clear();
            for (int32_t u : buckets[k]) {
                bool current = (size_t)(dist[u] / delta) == k;
                if (current && !in_frontier[u]) {
                    in_frontier[u] = 1;
                    frontier.push_back(u);
                }
            }
            buckets[k].clear();

            for (int32_t u : frontier) {
                in_frontier[u] = 0;
                if (!in_settled[u]) {
                    in_settled[u] = 1;
                    settled.push_back(u);
                }
            }

            relax_edges(frontier, true);
            stats.phases++;

            int local_more  = buckets[k].empty() ? 0 : 1;
            int global_more = 0;
            MPI_Allreduce(&local_more, &global_more, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (!global_more) break;
        }

        // heavy edges (w > delta) always land in a later bucket, so one pass
        // over everything settled in bucket k is enough
        relax_edges(settled, false);
        for (int32_t u : settled) in_settled[u] = 0;
    }

    // per-rank counters -> global totals
    long long local_counts[3]  = {stats.relaxations, stats.remote_requests, stats.bytes_sent};
    long long global_counts[3] = {0, 0, 0};
    MPI_Allreduce(local_counts, global_counts, 3, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    stats.relaxations     = global_counts[0];
    stats.remote_requests = global_counts[1];
    stats.bytes_sent      = global_counts[2];

    stats.runtime_seconds = timer.stop();
    return dist;
}

void print_delta_stats(const DeltaStats& s, int64_t delta) {
    std::cout << "\n=== Metrics: Delta-Stepping (delta = " << delta << ") ===\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Runtime:          " << s.runtime_seconds << " seconds\n";
    std::cout << "  Buckets:          " << s.buckets << "\n";
    std::cout << "  Light phases:     " << s.phases << "\n";
    std::cout << "  Relaxations:      " << s.relaxations << "\n";
    std::cout << "  Remote requests:  " << s.remote_requests << "\n";
    std::cout << "  Bytes sent:       " << s.bytes_sent << "\n";
    std::cout << "========================\n\n";
}
