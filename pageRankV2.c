/*
 * pageRankV2.c
 * Parallel PageRank on large sparse graphs using MPI + CSR storage.
 *
 * Extends the dense PageRank implementation to handle arbitrarily large web
 * graphs supplied as edge-list files.  The graph is loaded into Compressed
 * Sparse Row (CSR) format so only non-zero entries consume memory, making
 * this suitable for real-world graphs with millions of nodes.
 *
 * The stochastic Google matrix is never materialised explicitly; instead,
 * each iteration applies the dangling-node correction and teleportation
 * analytically, operating only on stored non-zeros.  Row blocks are
 * distributed across MPI processes and reassembled with MPI_Allgatherv.
 *
 * Input file format:
 *   Line 1:  <N>          (number of nodes, 0-indexed)
 *   Lines 2+: <src> <dst>  (directed edge from src to dst)
 *
 * Usage:
 *   mpiexec -n <num_procs> ./pageRankV2 <graph_file> <epsilon>
 *   Example: mpiexec -n 4 ./pageRankV2 web.txt 1e-6
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define ALPHA 0.85  // Damping factor (standard PageRank value)
int SIZE;           // Number of nodes; set at runtime from the graph file

/*
 * CSR (Compressed Sparse Row) matrix representation.
 * For PageRank we index by DESTINATION row, so col_ind stores source nodes.
 * This lets us compute the in-link contribution for each page efficiently.
 */
typedef struct {
    int    *col_ind;  // Source node for each non-zero entry
    double *values;   // Transition weight: 1 / out-degree of source node
    int    *row_ptr;  // row_ptr[i]..row_ptr[i+1]-1 = non-zeros for row i
    int     nnz;      // Total number of non-zero entries
} CSRMatrix;

double *a1;      // Dangling-node indicator: a1[j]=1 if node j has no out-edges
double *pi_old;  // PageRank vector from the previous iteration
double *pi_new;  // PageRank vector being computed in the current iteration

/*
 * load_CSR
 * Read a directed graph from an edge-list file and build a CSR matrix
 * indexed by destination node, so each row i contains the sources that
 * link INTO node i weighted by their out-degree.
 */
CSRMatrix load_CSR(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file\n"); exit(1);
    }

    fscanf(file, "%d", &SIZE);
    int *out_degree = calloc(SIZE, sizeof(int));
    int *in_degree = calloc(SIZE, sizeof(int));
    int src, dst, edge_count = 0;

    // First pass: count degrees to determine CSR row sizes
    while (fscanf(file, "%d %d", &src, &dst) == 2) {
        out_degree[src]++;
        in_degree[dst]++;
        edge_count++;
    }

    CSRMatrix csr;
    csr.nnz = edge_count;
    csr.col_ind = malloc(edge_count * sizeof(int));
    csr.values = malloc(edge_count * sizeof(double));
    csr.row_ptr = malloc((SIZE + 1) * sizeof(int));

    // Mark dangling nodes (no outgoing edges) for the teleportation term
    a1 = calloc(SIZE, sizeof(double));
    for (int i = 0; i < SIZE; i++) {
        if (out_degree[i] == 0) a1[i] = 1.0;
    }

    rewind(file);
    fscanf(file, "%d", &SIZE);

    // Build row pointers from in-degree counts
    int *current_pos = calloc(SIZE, sizeof(int));
    csr.row_ptr[0] = 0;
    for (int i = 1; i <= SIZE; i++) {
        csr.row_ptr[i] = csr.row_ptr[i - 1] + in_degree[i - 1];
    }

    // Second pass: populate col_ind and values indexed by destination
    while (fscanf(file, "%d %d", &src, &dst) == 2) {
        int idx = csr.row_ptr[dst] + current_pos[dst];
        csr.col_ind[idx] = src;
        csr.values[idx] = 1.0 / out_degree[src];
        current_pos[dst]++;
    }

    free(out_degree);
    free(in_degree);
    free(current_pos);
    fclose(file);
    return csr;
}

/*
 * multiply_CSR_block
 * Compute one power-iteration step for a contiguous block of rows.
 * Applies the full Google Matrix formula without materialising G:
 *   π_new[i] = α·Σ_j H[i][j]·π_old[j]  +  α·(1/N)·Σ_j a1[j]·π_old[j]
 *              + (1-α)·(1/N)
 * The dangling sum and teleportation constant are computed once per call,
 * not once per non-zero, for efficiency.
 */
void multiply_CSR_block(CSRMatrix csr, double *pi_old, double *pi_new,
                        int start_row, int end_row, double inv_n) {
    // Accumulate contribution from dangling nodes once for all rows
    double dangling_sum = 0.0;
    for (int j = 0; j < SIZE; j++) {
        if (a1[j] > 0) dangling_sum += pi_old[j];
    }

    // Teleportation constant is identical for every row
    double teleport = (1.0 - ALPHA) * inv_n + ALPHA * inv_n * dangling_sum;

    for (int i = start_row; i < end_row; i++) {
        double sum = 0.0;
        for (int idx = csr.row_ptr[i]; idx < csr.row_ptr[i + 1]; idx++) {
            sum += ALPHA * csr.values[idx] * pi_old[csr.col_ind[idx]];
        }
        pi_new[i - start_row] = sum + teleport;
    }
}

int main(int argc, char **argv) {
    int num_proc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 3) {
        if (rank == 0) {
            printf("Usage: mpiexec -n <num_procs> ./pageRankV2 <file> <epsilon>\n");
        }
        MPI_Finalize();
        return 0;
    }

    double epsilon = atof(argv[2]);
    double start_time = MPI_Wtime();
    CSRMatrix csr = load_CSR(argv[1]);
    double inv_n = 1.0 / SIZE;

    // Initialize rank vector to uniform distribution
    pi_old = calloc(SIZE, sizeof(double));
    pi_new = calloc(SIZE, sizeof(double));
    for (int i = 0; i < SIZE; i++) pi_old[i] = inv_n;

    // Distribute rows across processes; remainder rows go to lower-ranked processes
    int rows_per_proc = SIZE / num_proc;
    int remainder = SIZE % num_proc;
    int start = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int count = rows_per_proc + (rank < remainder ? 1 : 0);
    int end = start + count;

    int *recvcounts = malloc(num_proc * sizeof(int));
    int *displs = malloc(num_proc * sizeof(int));
    for (int p = 0; p < num_proc; p++) {
        recvcounts[p] = rows_per_proc + (p < remainder ? 1 : 0);
        displs[p] = p * rows_per_proc + (p < remainder ? p : remainder);
    }

    double *local_values = malloc(count * sizeof(double));
    double diff = 1.0;
    int iterations = 0;
    double rank_time = MPI_Wtime();

    // Power iteration: converge when L1 norm of update drops below epsilon
    while (diff > epsilon) {
        multiply_CSR_block(csr, pi_old, local_values, start, end, inv_n);
        MPI_Allgatherv(local_values, count, MPI_DOUBLE, pi_new, recvcounts,
                       displs, MPI_DOUBLE, MPI_COMM_WORLD);

        // Re-normalise to correct floating-point drift
        double total = 0.0;
        for (int i = 0; i < SIZE; i++) {
            total += pi_new[i];
        }
        for (int i = 0; i < SIZE; i++) {
            pi_new[i] /= total;
        }

        diff = 0.0;
        for (int i = 0; i < SIZE; i++) {
            diff += fabs(pi_new[i] - pi_old[i]);
        }

        for (int i = 0; i < SIZE; i++) {
            pi_old[i] = pi_new[i];
        }
        iterations++;
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        double total = 0.0;
        for (int i = 0; i < SIZE; i++) {
            total += pi_old[i];
        }

        printf("--------------------------------------------\n");
        printf("------------ Reached Convergence -----------\n");
        printf("Processes used:        %d\n",   num_proc);
        printf("Nodes:                 %d\n",   SIZE);
        printf("Epsilon value:         %g\n",   epsilon);
        printf("Converged in:          %d iterations\n", iterations);
        printf("Runtime (full):        %f seconds\n", end_time - start_time);
        printf("Runtime (rank-only):   %f seconds\n", end_time - rank_time);

        if (fabs(total - 1.0) < 1e-6) {
            printf("PageRank vector sums to 1.0 — check passed.\n");
        } else {
            printf("PageRank vector sum check FAILED (sum = %f).\n", total);
        }
    }

    free(csr.col_ind);
    free(csr.values);
    free(csr.row_ptr);
    free(a1);
    free(pi_old);
    free(pi_new);
    free(local_values);
    free(recvcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
