# NGS-MPI: Distributed Graph Algorithms with MPI

Distributed graph algorithms in C++17 and OpenMPI: graphs are partitioned across MPI ranks, which run leader election and single-source shortest paths.

> **Work in progress:** the shortest-path code is being reworked from a replicated-graph Dijkstra baseline into distributed delta-stepping on large synthetic graphs.

---
## Quickstart

Run the project end-to-end using the included graph:

```bash
# build
cmake -S mpi_runtime -B build
cmake --build build

# partition graph
python3 tools/partition/partition.py outputs/graph.json \
  --ranks 4 --out outputs/part.json --strategy contiguous

# run leader election
mpirun -n 4 ./build/ngs_mpi \
  --graph outputs/graph.json \
  --part outputs/part.json \
  --algo leader --rounds 200

# run dijkstra
mpirun -n 4 ./build/ngs_mpi \
  --graph outputs/graph.json \
  --part outputs/part.json \
  --algo dijkstra-baseline --source 0
```

### Second Graph

A second small graph is included at `outputs/graph2.json`. Run it the same way:

```bash
python3 tools/partition/partition.py outputs/graph2.json \
  --ranks 4 --out outputs/part2.json --strategy contiguous

mpirun -n 4 ./build/ngs_mpi --graph outputs/graph2.json \
  --part outputs/part2.json --algo leader --rounds 200

mpirun -n 4 ./build/ngs_mpi --graph outputs/graph2.json \
  --part outputs/part2.json --algo dijkstra-baseline --source 0
```

---

## Reproducibility

### Small Graphs

`outputs/graph.json` (501 nodes) and `outputs/graph2.json` are fixed, checked-in
graphs. All v1 experiments in `REPORT.md` were run on `outputs/graph.json`.

### Partitioning

Graph partitioning **is reproducible**:

- The partitioning tool supports a `--seed` flag
- Default seed is `42`
- Given the same input graph and seed, partition outputs are deterministic

---
## Pipeline

```
graph.json
    ↓
partition.py → part.json
    ↓
C++17 MPI runtime (ngs_mpi)
    - Leader Election (FloodMax)
    - Dijkstra Shortest Paths
    ↓
Metrics output
```

---

## Language Stack

* Graph partitioning: Python 3
* Core MPI runtime and algorithms: C++17 + OpenMPI

The distributed algorithms (leader election and Dijkstra) are implemented in C++17 using OpenMPI.

---

## Prerequisites

Install the following on Ubuntu or WSL2:

```bash
# Python
sudo apt install python3

# OpenMPI (required — do not use MPICH on WSL2)
sudo apt install libopenmpi-dev openmpi-bin

# Build tools
sudo apt install cmake g++

# Testing
sudo apt install libgtest-dev
```

---

## Repository Structure

```
tools/
  partition/
    partition.py
mpi_runtime/
  src/
  include/
  CMakeLists.txt
experiments/
  run_experiments.sh
outputs/
  graph.json
REPORT.md
README.md
```

---

## Build

```bash
cmake -S mpi_runtime -B build
cmake --build build
```

Binaries produced in `build/`:

* `ngs_mpi` — main runtime
* `unit_tests` — unit tests
* `mpi_tests` — MPI tests

---

## Running the Pipeline

### Step 1: Partition the Graph

```bash
# Contiguous
python3 tools/partition/partition.py outputs/graph.json \
  --ranks 4 --out outputs/part.json --strategy contiguous

# Random
python3 tools/partition/partition.py outputs/graph.json \
  --ranks 4 --out outputs/part.json --strategy random
```

---

### Step 2: Run Algorithms

```bash
# Leader election
mpirun -n 4 ./build/ngs_mpi \
  --graph outputs/graph.json \
  --part outputs/part.json \
  --algo leader --rounds 200

# Dijkstra
mpirun -n 4 ./build/ngs_mpi \
  --graph outputs/graph.json \
  --part outputs/part.json \
  --algo dijkstra-baseline --source 0
```

Oversubscribe if needed:

```bash
mpirun --oversubscribe -n 8 ./build/ngs_mpi ...
```

---

## Distributed Execution Model

* Each MPI rank loads the full graph
* Each rank owns a subset of nodes defined by the partition
* Execution follows a bulk-synchronous model using MPI collectives

### Leader Election (FloodMax)

* Each node maintains a candidate ID
* Each round:

  * Local updates propagate values
  * `MPI_Allreduce (MAX)` synchronizes global state
* Terminates when no changes occur globally

### Dijkstra

* Each iteration:

  * Each rank computes its local minimum tentative distance
  * Global minimum (distance and node id) selected with one `MPI_Allreduce` using `MPI_MINLOC`
  * All ranks relax edges from that node
* Continues until all nodes are settled

---

## Input Validation

The runtime validates all inputs before execution:

* Graph file must be valid JSON with consistent node count
* Partition file must:

  * Assign every node exactly once
  * Use valid ranks in `[0, num_ranks)`
  * Match MPI world size
* CLI arguments:

  * `--algo` must be `leader` or `dijkstra-baseline`
  * `--rounds` must be positive
  * `--source` must be within node range

Invalid inputs produce descriptive errors and exit cleanly.

---

## Metrics

The runtime reports:

* Runtime (seconds)
* Iterations
* Allreduce calls (logical communication steps)
* Approximate bytes sent

Note:
MPI collectives do not correspond directly to point-to-point messages.
We approximate communication cost as **O(ranks)** per collective for comparison purposes.

---

## Running Tests

```bash
# Unit tests
cd mpi_runtime && ../build/unit_tests

# MPI tests
cd mpi_runtime && mpirun -n 4 ../build/mpi_tests
```

All 11 tests should pass (7 unit, 4 MPI).


---

## Reproducing Experiments

```bash
./experiments/run_experiments.sh
```

Outputs saved to `experiments/results.txt`.

Experiments include:

* Contiguous vs random partitioning (4 ranks)
* Scalability across 1, 2, 4, 8 ranks

---

## Assumptions and Limitations

* The included JSON graphs are undirected (every edge has a reverse edge)
* Graph is assumed to be connected
* Dijkstra requires non-negative edge weights
* Full graph is replicated on all ranks (not memory optimized)
* Communication uses synchronous collectives (no async overlap)

---

## Notes

* `outputs/graph.json` is included for reproducibility
* Use OpenMPI, MPICH may fail on WSL2
* Leader election correctness depends on graph connectivity
