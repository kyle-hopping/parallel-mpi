/*
 * primeGap.c
 * Parallel largest-prime-gap finder using MPI and a segmented sieve.
 *
 * Finds the largest gap between consecutive prime numbers within a
 * user-specified range [start, end].  The search space is divided evenly
 * across MPI processes using domain decomposition.  Each process runs a
 * segmented Sieve of Eratosthenes on its sub-range, processing the range
 * in fixed-size blocks to keep memory usage constant regardless of range
 * size.
 *
 * Boundary gaps — where one prime falls in one process's range and its
 * successor falls in the next — are handled explicitly via point-to-point
 * communication between adjacent ranks.  Local maxima are then reduced to
 * a global maximum on rank 0 using MPI_Gather.
 *
 * Usage:
 *   mpirun -n <num_procs> ./primeGap <start> <end>
 *   Example: mpirun -n 4 ./primeGap 2 1000000000
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#define BLOCK_SIZE 262144  // Processing chunk size in bytes (256 KB)

// Holds the gap size and the two primes that bound it
typedef struct {
    unsigned long long gap;
    unsigned long long p1;
    unsigned long long p2;
} GapResult;

/*
 * sieve
 * Generate all primes up to `limit` using the classical Sieve of Eratosthenes.
 * These "seed primes" are later used to mark composites in the segmented sieve.
 *
 * Returns a heap-allocated array of primes; sets *count to the array length.
 * Caller is responsible for freeing the returned pointer.
 */
unsigned long long *sieve(unsigned long long limit, int *count) {
    if (limit < 2) { *count = 0; return NULL; }

    // Boolean sieve array: 1 = prime candidate, 0 = composite
    char *is_prime = (char *)calloc(limit + 1, sizeof(char));
    memset(is_prime, 1, limit + 1);
    is_prime[0] = is_prime[1] = 0;

    for (unsigned long long p = 2; p * p <= limit; p++) {
        if (is_prime[p]) {
            for (unsigned long long i = p * p; i <= limit; i += p) {
                is_prime[i] = 0;
            }
        }
    }

    int c = 0;
    for (unsigned long long p = 2; p <= limit; p++) {
        if (is_prime[p]) c++;
    }

    unsigned long long *primes = malloc(c * sizeof(unsigned long long));
    int idx = 0;
    for (unsigned long long p = 2; p <= limit; p++) {
        if (is_prime[p]) primes[idx++] = p;
    }
    free(is_prime);
    *count = c;
    return primes;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: %s <start> <end>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    unsigned long long global_start = strtoull(argv[1], NULL, 10);
    unsigned long long global_end = strtoull(argv[2], NULL, 10);
    if (global_start < 2) global_start = 2;

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Divide the range evenly; the last rank absorbs any leftover
    unsigned long long total_range = global_end - global_start + 1;
    unsigned long long range_per_rank = total_range / size;
    unsigned long long my_start = global_start + (rank * range_per_rank);
    unsigned long long my_end = (rank == size - 1) ? global_end : (my_start + range_per_rank - 1);

    if (rank == 0) {
        printf("Searching [%llu, %llu] with %d processors\n\n", global_start, global_end, size);
    }

    /*
     * Generate seed primes up to sqrt(global_end) on rank 0, then broadcast.
     * Any composite in the range has at least one prime factor <= sqrt(end),
     * so these seeds are sufficient to sieve the entire range.
     */
    unsigned long long sqrt_end = (unsigned long long)sqrt((double)global_end) + 1;
    int seed_count = 0;
    unsigned long long *seeds = NULL;

    if (rank == 0) {
        printf("Generating seed primes up to %llu...\n", sqrt_end);
        seeds = sieve(sqrt_end, &seed_count);
        printf("Generated %d seed primes\n\n", seed_count);
    }

    MPI_Bcast(&seed_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        seeds = malloc(seed_count * sizeof(unsigned long long));
    }
    MPI_Bcast(seeds, seed_count, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // Per-process segmented sieve state
    char *block = malloc(BLOCK_SIZE * sizeof(char));
    GapResult local_max = {0, 0, 0};
    unsigned long long last_prime = 0;
    unsigned long long first_prime_in_rank = 0;
    unsigned long long last_prime_in_rank = 0;

    // Process this rank's sub-range one BLOCK_SIZE chunk at a time
    for (unsigned long long curr = my_start; curr <= my_end; curr += BLOCK_SIZE) {
        unsigned long long limit    = curr + BLOCK_SIZE - 1;
        if (limit > my_end) {
            limit   = my_end;
        }
        unsigned long long block_sz = limit - curr + 1;
        memset(block, 1, block_sz);  // Assume every number is prime initially

        // Cross off multiples of each seed prime within this block
        for (int i = 0; i < seed_count; i++) {
            unsigned long long p  = seeds[i];
            unsigned long long p2 = p * p;
            if (p2 > limit) break;  // No seed prime can affect numbers beyond p^2

            unsigned long long start_index;
            if (p2 >= curr) {
                start_index = p2 - curr;
            } else {
                unsigned long long rem = curr % p;
                start_index = (rem == 0) ? 0 : (p - rem);
            }

            for (unsigned long long j = start_index; j < block_sz; j += p) {
                block[j] = 0;
            }
        }

        // Scan the sieved block and record gaps between consecutive primes
        for (unsigned long long j = 0; j < block_sz; j++) {
            if (!block[j]) {
                continue;
            }
            unsigned long long p = curr + j;

            if (p < 2) {
                continue;
            }

            if (first_prime_in_rank == 0) {
                first_prime_in_rank = p;
            }

            if (last_prime > 0) {
                unsigned long long gap = p - last_prime;
                if (gap > local_max.gap) {
                    local_max.gap = gap;
                    local_max.p1 = last_prime;
                    local_max.p2 = p;
                }
            }
            last_prime = p;
            last_prime_in_rank = p;
        }
    }

    /*
     * Handle cross-boundary gaps: each process sends its last prime to its
     * right neighbour, which checks whether the gap across the boundary
     * exceeds its local maximum.
     */
    if (rank < size - 1 && last_prime_in_rank > 0) {
        MPI_Send(&last_prime_in_rank, 1, MPI_UNSIGNED_LONG_LONG, rank + 1, 0, MPI_COMM_WORLD);
    }

    if (rank > 0) {
        unsigned long long prev_last_prime;
        MPI_Recv(&prev_last_prime, 1, MPI_UNSIGNED_LONG_LONG, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (first_prime_in_rank > 0 && prev_last_prime > 0) {
            unsigned long long boundary_gap = first_prime_in_rank - prev_last_prime;
            if (boundary_gap > local_max.gap) {
                local_max.gap = boundary_gap;
                local_max.p1 = prev_last_prime;
                local_max.p2 = first_prime_in_rank;
            }
        }
    }

    // Gather all local maxima to rank 0 for final reduction
    GapResult *gathered = NULL;
    if (rank == 0) {
        gathered = malloc(size * sizeof(GapResult));
    }

    MPI_Gather(&local_max, sizeof(GapResult), MPI_BYTE, gathered, sizeof(GapResult), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        // Linear scan to find the global maximum across all processes
        GapResult global_max = gathered[0];
        int winner = 0;
        for (int i = 1; i < size; i++) {
            if (gathered[i].gap > global_max.gap) {
                global_max = gathered[i];
                winner = i;
            }
        }

        printf("=====================================\n");
        printf("RESULTS\n");
        printf("=====================================\n");
        printf("Range:           [%llu, %llu]\n", global_start, global_end);
        printf("Processors:      %d\n", size);
        printf("Max gap:         %llu\n", global_max.gap);
        printf("Between primes:  %llu and %llu\n", global_max.p1, global_max.p2);
        printf("Found by rank:   %d\n", winner);
        printf("Time:            %.6f seconds\n", end_time - start_time);
        printf("=====================================\n");

        free(gathered);
    }

    free(seeds);
    free(block);
    MPI_Finalize();
    return 0;
}
