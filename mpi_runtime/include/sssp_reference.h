#pragma once
#include "csr.h"
#include <cstdint>
#include <vector>

// Distance value for nodes the source cannot reach
const int64_t SSSP_INF = INT64_MAX / 4;

// Single-process binary-heap Dijkstra over the whole graph.
// Used as the answer key when verifying distributed results.
std::vector<int64_t> sequential_sssp(const CsrGraph& g, int32_t source);
