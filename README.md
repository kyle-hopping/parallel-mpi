# ⚡ Parallel Computing with MPI

A collection of parallel algorithms implemented in C using the **Message Passing Interface (MPI)**.  
Each program tackles a classic problem in parallel computing — from web graph ranking to prime number theory — and is designed to scale efficiently across multiple processors.

---

## 📂 Projects

### 🌐 PageRank (Dense) — `pageRank.c`
An MPI-parallel implementation of Google's **PageRank algorithm** on a fixed 15-node web graph.

- Builds the full Google Matrix **G = αS + (1−α)(1/N)E**, including dangling-node correction
- Distributes matrix rows across processes and uses **power iteration** until convergence
- Assembles the global rank vector each iteration with `MPI_Allgatherv`
- Prints the final ranked leaderboard of pages by score

```bash
mpiexec -n 4 ./pageRank 1e-6
```

---

### 🌐 PageRank (Sparse / Large Graphs) — `pageRankV2.c`
Extends the dense PageRank to handle **real-world, large-scale web graphs** loaded from edge-list files.

- Stores the graph in **Compressed Sparse Row (CSR)** format — only non-zero entries are kept in memory
- Applies dangling-node correction and teleportation analytically, never materialising the full Google Matrix
- Scales to graphs with millions of nodes where the dense version would be infeasible

```bash
# Input file format:
# Line 1: <N>  (number of nodes)
# Lines 2+: <src> <dst>  (directed edge)

mpiexec -n 4 ./pageRankV2 web.txt 1e-6
```

---

### 🔀 Parallel Merge — `parallelMerge.c`
Merges two large sorted arrays in parallel using the **co-rank algorithm**.

- Each process independently determines its exact slice of both input arrays using a **dual binary search (co-rank)**, achieving perfectly balanced work with no communication needed at partition time
- Each process performs a standard sequential merge on its slice, then all results are gathered with `MPI_Gatherv`
- Verifies correctness by checking the final array is globally sorted

```bash
mpirun -n 4 ./parallelMerge 10000 5000 42
#                             ^len1  ^len2  ^seed
```

---

### 🔢 Prime Gap Finder — `primeGap.c`
Finds the **largest gap between consecutive primes** in a given range using a parallel segmented sieve.

- Domain-decomposes the search range across processes; each runs a **segmented Sieve of Eratosthenes** in fixed-size 256 KB blocks (constant memory, arbitrary range size)
- Handles **cross-boundary gaps** explicitly with point-to-point communication between adjacent ranks
- Local maxima are gathered on rank 0 for the final reduction

```bash
mpirun -n 4 ./primeGap 2 1000000000
```

---

## 🛠️ Building

All programs require an MPI implementation (e.g. [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/)) and a C compiler.

```bash
# PageRank (dense)
mpicc -o pageRank pageRank.c -lm

# PageRank (sparse / large graphs)
mpicc -o pageRankV2 pageRankV2.c -lm

# Parallel Merge
mpicc -o parallelMerge parallelMerge.c -lm

# Prime Gap Finder
mpicc -o primeGap primeGap.c -lm
```

---

## 🔑 Key Concepts

| Concept | Where it appears |
|---|---|
| Domain decomposition | PageRank, Prime Gap |
| Sparse matrix (CSR) | PageRankV2 |
| Power iteration | PageRank, PageRankV2 |
| Co-rank binary search | Parallel Merge |
| Segmented sieve | Prime Gap |
| `MPI_Allgatherv` | PageRank, PageRankV2 |
| `MPI_Gatherv` | Parallel Merge |
| Point-to-point messaging | Prime Gap (boundary handling) |
| Load balancing | All programs |

---

## 📊 Performance Notes

- All programs use `MPI_Wtime()` to report wall-clock runtimes
- PageRank reports both total runtime and rank-computation-only runtime
- Parallel Merge reports elements processed per second
- Prime Gap reports time-to-solution across all tested process counts

---

## 🖥️ Development Environment

All programs were developed and tested on **[SciNet](https://research.utoronto.ca/training-resources-facilities/scinet)**, the high-performance computing facility at the University of Toronto. SciNet provides access to large-scale parallel computing infrastructure, making it an ideal environment for benchmarking MPI workloads across many processes and nodes.

---

## 🧰 Requirements

- C99 or later
- MPI (OpenMPI ≥ 4.0 or MPICH ≥ 3.3 recommended)
- `-lm` flag for `<math.h>`

---

## 👤 Author

**[Your Name]**  
[Your University / Program]  
[Your LinkedIn / GitHub / Email]
