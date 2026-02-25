#include <gtest/gtest.h>
#include <mpi.h>
#include "ops_mpi.h"

TEST(allreduce_test, sum_ints) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int send = rank + 1;
    int recv;
    
    my_allreduce(&send, &recv, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    int expected = 0;
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for (int i = 0; i < size; i++) expected += (i + 1);
    
    EXPECT_EQ(recv, expected);
}
