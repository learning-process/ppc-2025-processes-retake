#include <gtest/gtest.h>
#include <vector>
#include "marov_allreduce/mpi/include/ops_mpi.h"

TEST(marov_allreduce_mpi, sum_ints) {
    const int NUM_PROCS = 7;
    std::vector<int> sendValues(NUM_PROCS);
    std::vector<int> recvValues(NUM_PROCS);

    for(int i = 0; i < NUM_PROCS; i++) {
        sendValues[i] = i + 1;
    }

    for(int rank = 0; rank < NUM_PROCS; rank++) {
        MPI_Comm comm(rank, NUM_PROCS);
        int sendData = sendValues[rank];
        int recvData = 0;

        my_allreduce(&sendData, &recvData, 1, MPI_INT, MPI_SUM, &comm);
        recvValues[rank] = recvData;
    }

    int expected = 0;
    for(int v : sendValues) expected += v;
    for(int r : recvValues) {
        EXPECT_EQ(r, expected);
    }
}

TEST(marov_allreduce_mpi, max_floats) {
    const int NUM_PROCS = 7;
    std::vector<float> sendValues(NUM_PROCS);
    std::vector<float> recvValues(NUM_PROCS);

    for(int i = 0; i < NUM_PROCS; i++) {
        sendValues[i] = i * 1.5f;
    }

    for(int rank = 0; rank < NUM_PROCS; rank++) {
        MPI_Comm comm(rank, NUM_PROCS);
        float sendData = sendValues[rank];
        float recvData = 0;

        my_allreduce(&sendData, &recvData, 1, MPI_FLOAT, MPI_MAX, &comm);
        recvValues[rank] = recvData;
    }

    float expected = 0;
    for(float v : sendValues) if(v > expected) expected = v;
    for(float r : recvValues) {
        EXPECT_EQ(r, expected);
    }
}

TEST(marov_allreduce_mpi, min_doubles) {
    const int NUM_PROCS = 7;
    const int ARRAY_SIZE = 3;
    std::vector<std::vector<double>> sendValues(NUM_PROCS, std::vector<double>(ARRAY_SIZE));
    std::vector<std::vector<double>> recvValues(NUM_PROCS, std::vector<double>(ARRAY_SIZE));

    for(int i = 0; i < NUM_PROCS; i++) {
        sendValues[i][0] = i + 1.0;
        sendValues[i][1] = i * 2.0;
        sendValues[i][2] = i + 0.5;
    }

    for(int rank = 0; rank < NUM_PROCS; rank++) {
        MPI_Comm comm(rank, NUM_PROCS);
        double sendData[ARRAY_SIZE];
        double recvData[ARRAY_SIZE];

        for(int j = 0; j < ARRAY_SIZE; j++) {
            sendData[j] = sendValues[rank][j];
            recvData[j] = 0;
        }

        my_allreduce(sendData, recvData, ARRAY_SIZE, MPI_DOUBLE, MPI_MIN, &comm);

        for(int j = 0; j < ARRAY_SIZE; j++) {
            recvValues[rank][j] = recvData[j];
        }
    }

    std::vector<double> expected(ARRAY_SIZE, 1e9);
    for(int i = 0; i < NUM_PROCS; i++) {
        for(int j = 0; j < ARRAY_SIZE; j++) {
            if(sendValues[i][j] < expected[j]) {
                expected[j] = sendValues[i][j];
            }
        }
    }

    for(int rank = 0; rank < NUM_PROCS; rank++) {
        for(int j = 0; j < ARRAY_SIZE; j++) {
            EXPECT_EQ(recvValues[rank][j], expected[j]);
        }
    }
}
