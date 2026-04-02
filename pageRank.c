/*
 * Parallel PageRank computation using MPI.
 *
 * Implements the Google PageRank algorithm on a fixed 15-node web graph.
 * Builds the stochastic Google Matrix G = α·S + (1-α)·(1/N)·E, where S is
 * the dangling-corrected hyperlink matrix and α (ALPHA) is the damping factor.
 * Power iteration runs until the L1 norm of successive rank vectors falls
 * below a user-supplied epsilon.  Row computation is distributed across MPI
 * processes; results are assembled with MPI_Allgatherv each iteration.
 *
 * Usage:
 *   mpiexec -n <num_procs> ./pageRank <epsilon>
 *   Example: mpiexec -n 4 ./pageRank 1e-6
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define SIZE  15     // Number of nodes in the web graph
#define ALPHA 0.85   // Damping factor (standard PageRank value)

// Hyperlink matrix H: H[i][j] = probability of jumping from page j to page i
double page_rank_matrix[SIZE][SIZE] = {
    {0, 1.0 / 14, 1.0 / 14, 1.0 / 14, 1.0 / 14, 1.0 / 14, 1.0 / 14, 1.0 / 14, 1.0 / 14, 1.0 / 14, 1.0 / 14, 1.0 / 14, 1.0 / 14, 1.0 / 14, 1.0 / 14},
    {1.0 / 2, 0, 1.0 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1.0 / 3, 1.0 / 3, 0, 0, 1.0 / 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1.0 / 2, 0, 1.0 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1.0 / 8, 0, 1.0 / 8, 1.0 / 8, 1.0 / 8, 0, 0, 1.0 / 8, 0, 1.0 / 8, 0, 0, 0, 1.0 / 8, 1.0 / 8},
    {1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7, 0, 0, 0, 1.0 / 7, 0, 1.0 / 7, 0, 0, 0, 0},
    {1.0 / 4, 0, 0, 0, 0, 1.0 / 4, 0, 0, 0, 1.0 / 4, 0, 0, 0, 1.0 / 4, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1.0 / 5, 0, 1.0 / 5, 1.0 / 5, 0, 1.0 / 5, 0, 1.0 / 5, 0, 0, 0, 0, 0, 0, 0},
    {1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1.0 / 4, 0, 0, 1.0 / 4, 0, 0, 0, 0, 1.0 / 4, 0, 0, 0, 1.0 / 4, 0, 0},
    {1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1.0 / 4, 0, 1.0 / 4, 0, 0, 0, 0, 1.0 / 4, 1.0 / 4, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

// Dangling node indicator: a1[j] = 1 if page j has no outgoing links
double a1[SIZE] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1};

// Print the raw PageRank score vector
void print_vector(double *v, int n) {
    printf("\nPageRank vector:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", v[i]);
    }
    printf("\n");
}

// Print a SIZE×SIZE matrix with a label
void print_matrix(const char *name, double M[SIZE][SIZE]) {
    printf("\n%s:\n", name);
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%f ", M[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Sort pages by score and print a ranked leaderboard
void print_rankings(double *v, int n) {
    struct Pair { double score; int page; };
    struct Pair arr[n];

    for (int i = 0; i < n; i++) {
        arr[i].score = v[i];
        arr[i].page  = i + 1;
    }

    // Simple selection sort — n is small (15), so O(n^2) is fine
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (arr[j].score > arr[i].score) {
                struct Pair tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
            }
        }
    }

    printf("PageRank Rankings (Highest to Lowest):\n");
    for (int i = 0; i < n; i++) {
        printf("Rank %2d: Page %2d  —  %f\n", i + 1, arr[i].page, arr[i].score);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    int num_proc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: mpiexec -n <num_procs> ./pageRank <epsilon>\n");
        }
        MPI_Finalize();
        return 0;
    }

    double start_time = MPI_Wtime();

    double epsilon = atof(argv[1]);
    double inv_n   = 1.0 / SIZE;

    // Build the stochastic matrix S and the full Google Matrix G
    double S_matrix[SIZE][SIZE];
    double G[SIZE][SIZE];
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            double S      = page_rank_matrix[i][j] + a1[j] * inv_n;
            S_matrix[i][j] = S;
            G[i][j]       = ALPHA * S + (1.0 - ALPHA) * inv_n;
        }
    }

    // Distribute rows evenly; handle remainder by giving extra rows to lower ranks
    int rows_per_proc = SIZE / num_proc;
    int remainder = SIZE % num_proc;
    int start = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int count = rows_per_proc + (rank < remainder ? 1 : 0);
    int end = start + count;

    // Initialize rank vector to uniform distribution
    double pi_old[SIZE], pi_new[SIZE];
    for (int i = 0; i < SIZE; i++) {
        pi_old[i] = inv_n;
    }

    // Precompute Allgatherv metadata
    int recvcounts[num_proc], displs[num_proc];
    for (int p = 0; p < num_proc; p++) {
        recvcounts[p] = rows_per_proc + (p < remainder ? 1 : 0);
        displs[p] = p * rows_per_proc + (p < remainder ? p : remainder);
    }

    double local_values[count];
    double diff = 1.0;
    int iterations = 0;
    double rank_time = MPI_Wtime();

    // Power iteration: each process computes its assigned rows of G·π 
    while (diff > epsilon) {
        for (int i = start; i < end; i++) {
            double sum = 0.0;
            for (int j = 0; j < SIZE; j++) {
                sum += G[i][j] * pi_old[j];
            }
            local_values[i - start] = sum;
        }

        MPI_Allgatherv(local_values, count, MPI_DOUBLE,
                       pi_new, recvcounts, displs, MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // Normalize to maintain a valid probability distribution
        double total = 0.0;
        for (int i = 0; i < SIZE; i++) {
            total += pi_new[i];
        }
        for (int i = 0; i < SIZE; i++) {
            pi_new[i] /= total;
        }

        // Check L1 convergence
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
        printf("--------------------------------------------\n");
        printf("---------------- Matrix Dump ---------------\n");
        print_matrix("Matrix H (Hyperlink Matrix)", page_rank_matrix);
        print_matrix("Matrix S (Dangling-corrected Matrix)", S_matrix);
        print_matrix("Matrix G (Google Matrix)", G);

        printf("--------------------------------------------\n");
        printf("------------ Reached Convergence -----------\n");
        printf("Processes used:      %d\n", num_proc);
        printf("Epsilon value:       %g\n", epsilon);
        printf("Converged in:        %d iterations\n", iterations);
        printf("Runtime (full):      %f seconds\n", end_time - start_time);
        printf("Runtime (rank-only): %f seconds\n", end_time - rank_time);

        print_vector(pi_old, SIZE);
        printf("\n");
        print_rankings(pi_old, SIZE);
    }

    MPI_Finalize();
    return 0;
}
