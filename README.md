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
#                           ^len1 ^len2 ^seed
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

## 📈 Benchmark Results

All benchmarks were run on the SHARCnet teaching cluster using OpenMPI 5.0.3.

### 🌐 PageRank — Dense (15-node WLU graph)

| ε | Iterations | Converged? |
|---|---|---|
| 10⁻¹ | 2 | Partial (8 pages tied) |
| 10⁻³ | 6 | ✅ Fully resolved |
| 10⁻⁵ | 13 | ✅ Fully resolved |
| 10⁻⁷ | 21 | ✅ Fully resolved |

Results were **identical across all process counts (P = 2–8)**, confirming correct parallel decomposition. Top-ranked page: **Laurier Library** (score: 0.0814). The two dangling nodes scored ~6.7× lower than the median, consistent with theory.

### 🌐 PageRank — Sparse (real-world SNAP datasets, ε = 10⁻⁵)

| Dataset | Nodes | P=2 (s) | P=16 (s) | Speedup |
|---|---|---|---|---|
| web-Stanford | 281,903 | 0.583 | 0.393 | 1.48× |
| web-BerkStan | 685,230 | 1.255 | 0.740 | 1.70× |
| web-Google | 875,713 | 1.760 | 1.228 | 1.43× |
| wiki-topcats | 1,791,489 | 3.771 | 1.809 | **2.08×** |

### 🔀 Parallel Merge — Co-rank vs. Baseline (1M vs 1M elements)

| Processes | Baseline (s) | Co-rank (s) | Speedup |
|---|---|---|---|
| 2 | 0.01089 | 0.00767 | 1.42× |
| 4 | 0.01056 | 0.00564 | 1.87× |
| 8 | 0.00723 | 0.00426 | **1.70×** |

Co-rank achieved up to **470 million elements/second** and **88–91% parallel efficiency** at 8 processes. On skewed arrays (1M vs 100K), the baseline degraded by 41% when scaling from 3→7 processes while co-rank continued to improve — up to a **3.33× speedup** over the baseline.

### 🔢 Prime Gap — Segmented Sieve (range: [2, 10⁹])

| Processes | Time (s) | Max Gap Found |
|---|---|---|
| 2 | 1.458 | 282 |
| 4 | 0.731 | 282 |
| 8 | **0.366** | 282 |

Largest gap in [2, 10⁹]: **282**, between primes 436,273,009 and 436,273,291. For the bonus range [2, 10¹²], the algorithm correctly found a gap of **540** and achieved a **3.76× speedup** from P=2 to P=8, with parallel efficiency above 94%.

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
