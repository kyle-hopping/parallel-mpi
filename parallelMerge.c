/*
 * parallelMerge.c
 * Parallel merge of two sorted arrays using MPI and the co-rank algorithm.
 *
 * Given two independently sorted integer arrays, the merged output is
 * partitioned into contiguous, non-overlapping segments that are assigned
 * to processes in rank order.  Each process independently determines its
 * exact slice of both input arrays using the co-rank function, performs a
 * standard sequential merge on that slice, then all partial results are
 * assembled on rank 0 with MPI_Gatherv.
 *
 * The co-rank binary search ensures a perfectly balanced workload regardless
 * of how values are distributed across the two arrays, giving O(log(m+n))
 * partition overhead and O((m+n)/P) merge work per process.
 *
 * Usage:
 *   mpirun -n <num_procs> ./parallelMerge <len1> <len2> <seed>
 *   Example: mpirun -n 4 ./parallelMerge 10000 5000 42
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define UPPER1 2000  // Maximum value for elements in array 1
#define LOWER1 1     // Minimum value for elements in array 1
#define UPPER2 2000  // Maximum value for elements in array 2
#define LOWER2 1     // Minimum value for elements in array 2

// Comparator for qsort — ascending integer order
int compare_ints(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

/*
 * rand_list
 * Fill ptr[0..length-1] with random integers in [lower_bound_num, upper_bound_num],
 * then sort the result in ascending order using qsort.
 */
int rand_list(int length, int *ptr, int lower, int upper, int seed) {
    srand(seed);
    for (int i = 0; i < length; i++) {
        ptr[i] = rand() % (upper - lower + 1) + lower;
    }
    qsort(ptr, length, sizeof(int), compare_ints);
    return 0;
}

/*
 * co_rank
 * Find the split point i in array A such that taking i elements from A and
 * (k-i) elements from B yields the first k elements of the merged sequence.
 *
 * Uses a dual binary search on both arrays simultaneously, converging in
 * O(log(min(m,n))) steps — much faster than a linear scan.
 *
 * Returns: i, the number of elements to take from A for this rank-k split.
 */
int co_rank(int k, int *A, int m, int *B, int n) {
    int i = (k < m) ? k : m;   // Start with as many elements from A as possible
    int j = k - i;
    int i_low = (k - n > 0) ? k - n : 0;
    int j_low = (k - m > 0) ? k - m : 0;
    int delta;

    while (1) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            // Too many from A: shift some elements to B's side
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = k - j;
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            // Too many from B: shift some elements to A's side
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = k - i;
        } else {
            break;  // Correct split found
        }
    }
    return i;
}

int main(int argc, char **argv) {
    int len1, len2, seed;
    int *list1, *list2;
    int max_proc, curr_proc;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &max_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_proc);

    if (argc != 4) {
        if (curr_proc == 0) {
            printf("Usage: mpirun -n <num_procs> ./parallelMerge <len1> <len2> <seed>\n");
        }
        MPI_Finalize();
        return 1;
    }

    len1 = atoi(argv[1]);
    len2 = atoi(argv[2]);
    seed = atoi(argv[3]);

    if (len1 <= 0 || len2 <= 0) {
        if (curr_proc == 0) {
            printf("Error: array lengths must be positive integers.\n");
        }
        MPI_Finalize();
        return 1;
    }

    list1 = (int *)malloc(len1 * sizeof(int));
    list2 = (int *)malloc(len2 * sizeof(int));

    if (!list1 || !list2) {
        fprintf(stderr, "Process %d: failed to allocate input arrays.\n", curr_proc);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Rank 0 generates both sorted arrays and prints a summary
    if (curr_proc == 0) {
        printf("========================================\n");
        printf("  PARALLEL ARRAY MERGE WITH MPI\n");
        printf("========================================\n\n");
        printf("Processes:      %d\n",    max_proc);
        printf("Array 1 length: %d\n",    len1);
        printf("Array 2 length: %d\n",    len2);
        printf("Total elements: %d\n",    len1 + len2);
        printf("Random seed:    %d\n\n",  seed);

        if (rand_list(len1, list1, LOWER1, UPPER1, seed) != 0 ||
            rand_list(len2, list2, LOWER2, UPPER2, seed + 1) != 0) {
                printf("Failed to generate random lists.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Print a representative sample: first 10 and last 5 elements
        printf("Array 1 sample (first 10): ");
        for (int i = 0; i < (len1 < 10 ? len1 : 10); i++) {
            printf("%d ", list1[i]);
        }
        if (len1 > 10) {
            printf("... ");
            for (int i = len1-5; i < len1; i++) {
                printf("%d ", list1[i]); 
            }
        }

        printf("\nArray 2 sample (first 10): ");
        for (int i = 0; i < (len2 < 10 ? len2 : 10); i++) {
            printf("%d ", list2[i]);
        }
        if (len2 > 10) {
            printf("... ");
            for (int i = len2-5; i < len2; i++) {
                printf("%d ", list2[i]);
            }
        }
        printf("\n\n");
    }

    // Broadcast both arrays to all processes before partitioning
    MPI_Bcast(list1, len1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(list2, len2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Partition boundary arrays: index[p] = start in each array for process p
    int *list1_index = (int *)malloc((max_proc + 1) * sizeof(int));
    int *list2_index = (int *)malloc((max_proc + 1) * sizeof(int));

    if (!list1_index || !list2_index) {
        fprintf(stderr, "Process %d: failed to allocate partition arrays.\n", curr_proc);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Rank 0 computes partition boundaries using co-rank and prints load info
    if (curr_proc == 0) {
        int total = len1 + len2;
        int target_per_proc = total / max_proc;

        printf("Load Balancing:\n");
        printf("Target elements per process: %d\n\n", target_per_proc);
        printf("%-10s %-15s %-20s %-20s %-15s\n",
               "Process", "Total Elements", "From Array 1", "From Array 2", "Balance");
        printf("------------------------------------------------------------------------\n");

        for (int p = 0; p <= max_proc; p++) {
            int k = (p * total) / max_proc;
            list1_index[p] = co_rank(k, list1, len1, list2, len2);
            list2_index[p] = k - list1_index[p];

            if (p < max_proc) {
                int next_k = ((p + 1) * total) / max_proc;
                int next_list1 = co_rank(next_k, list1, len1, list2, len2);
                int next_list2 = next_k - next_list1;
                int w1 = next_list1 - list1_index[p];
                int w2 = next_list2 - list2_index[p];
                double balance = (double)(w1 + w2) / target_per_proc * 100.0;
                printf("%-10d %-15d %-20d %-20d %.1f%%\n", p, w1+w2, w1, w2, balance);
            }
        }
        printf("\n");
    }

    // Share partition boundaries so every process can self-assign its slice
    MPI_Bcast(list1_index, max_proc + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(list2_index, max_proc + 1, MPI_INT, 0, MPI_COMM_WORLD);

    int list1_start = list1_index[curr_proc];
    int list1_end = list1_index[curr_proc + 1];
    int list2_start = list2_index[curr_proc];
    int list2_end = list2_index[curr_proc + 1];
    int local_len1 = list1_end - list1_start;
    int local_len2 = list2_end - list2_start;
    int merged_size = local_len1 + local_len2;

    int *merged_list = (int *)malloc(merged_size * sizeof(int));
    if (!merged_list && merged_size > 0) {
        fprintf(stderr, "Process %d: failed to allocate merged list.\n", curr_proc);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Sequential merge on this process's assigned slice of both arrays
    int i = list1_start, j = list2_start, k = 0;
    while (k < merged_size) {
        if (i < list1_end && j < list2_end) {
            merged_list[k++] = (list1[i] <= list2[j]) ? list1[i++] : list2[j++];
        } else if (i < list1_end) {
            merged_list[k++] = list1[i++];
        } else {
            merged_list[k++] = list2[j++];
        }
    }

    // Collect segment sizes, then gather all partial results into final array
    int *recv_counts = NULL;
    int *displacements = NULL;
    int *final_list = NULL;

    if (curr_proc == 0) {
        recv_counts = (int *)malloc(max_proc * sizeof(int));
        displacements = (int *)malloc(max_proc * sizeof(int));
    }

    MPI_Gather(&merged_size, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_size = 0;
    if (curr_proc == 0) {
        displacements[0] = 0;
        total_size = recv_counts[0];
        for (int p = 1; p < max_proc; p++) {
            displacements[p] = displacements[p - 1] + recv_counts[p - 1];
            total_size += recv_counts[p];
        }

        final_list = (int *)malloc(total_size * sizeof(int));
        if (!final_list) {
            fprintf(stderr, "Root: failed to allocate final list.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gatherv(merged_list, merged_size, MPI_INT, final_list, recv_counts,
                displacements, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (curr_proc == 0) {
        printf("Merge Results:\n");
        printf("Total merged elements: %d\n", total_size);

        // Verify correctness: the gathered result must be globally sorted
        int is_sorted = 1;
        for (int i = 1; i < total_size; i++) {
            if (final_list[i] < final_list[i - 1]) {
                is_sorted = 0;
                printf("ERROR: not sorted at index %d (%d > %d)\n", i, final_list[i - 1], final_list[i]);
                break;
            }
        }
        if (is_sorted) {
            printf("Result is correctly sorted.\n");
        }

        printf("\nMerged array sample (first 20): ");
        for (int i = 0; i < (total_size < 20 ? total_size : 20); i++) {
            printf("%d ", final_list[i]);
        }
        if (total_size > 20) {
            printf("... ");
            for (int i = total_size-10; i < total_size; i++) {
                printf("%d ", final_list[i]);
            }
        }
        printf("\n\n");

        printf("Performance:\n");
        printf("Execution time:      %.6f seconds\n", end_time - start_time);
        printf("Elements per second: %.2f million\n", (total_size / (end_time - start_time)) / 1e6);

        free(final_list);
        free(recv_counts);
        free(displacements);
    }

    free(list1);
    free(list2);
    free(list1_index);
    free(list2_index);
    free(merged_list);
    MPI_Finalize();
    return 0;
}
