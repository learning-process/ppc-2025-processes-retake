#include <gtest/gtest.h>
#include <vector>

#include "marov_allreduce/mpi/include/ops_mpi.h"

TEST(marovAllreduceMpi, sumInts) {
  const int kNumProcs = 7;
  std::vector<int> send_values(kNumProcs);
  std::vector<int> recv_values(kNumProcs);

  for (int i = 0; i < kNumProcs; i++) {
    send_values[i] = i + 1;
  }

  for (int rank = 0; rank < kNumProcs; rank++) {
    MPI_Comm comm(rank, kNumProcs);
    int send_data = send_values[rank];
    int recv_data = 0;

    my_allreduce(&send_data, &recv_data, 1, MPI_INT, MPI_SUM, &comm);
    recv_values[rank] = recv_data;
  }

  int expected = 0;
  for (int v : send_values) {
    expected += v;
  }
  for (int r : recv_values) {
    EXPECT_EQ(r, expected);
  }
}

TEST(marovAllreduceMpi, maxFloats) {
  const int kNumProcs = 7;
  std::vector<float> send_values(kNumProcs);
  std::vector<float> recv_values(kNumProcs);

  for (int i = 0; i < kNumProcs; i++) {
    send_values[i] = static_cast<float>(i) * 1.5F;
  }

  for (int rank = 0; rank < kNumProcs; rank++) {
    MPI_Comm comm(rank, kNumProcs);
    float send_data = send_values[rank];
    float recv_data = 0;

    my_allreduce(&send_data, &recv_data, 1, MPI_FLOAT, MPI_MAX, &comm);
    recv_values[rank] = recv_data;
  }

  float expected = 0;
  for (float v : send_values) {
    if (v > expected) {
      expected = v;
    }
  }
  for (float r : recv_values) {
    EXPECT_EQ(r, expected);
  }
}

TEST(marovAllreduceMpi, minDoubles) {
  const int kNumProcs = 7;
  const int kArraySize = 3;
  std::vector<std::vector<double>> send_values(
      kNumProcs, std::vector<double>(kArraySize));
  std::vector<std::vector<double>> recv_values(
      kNumProcs, std::vector<double>(kArraySize));

  for (int i = 0; i < kNumProcs; i++) {
    send_values[i][0] = i + 1.0;
    send_values[i][1] = i * 2.0;
    send_values[i][2] = i + 0.5;
  }

  for (int rank = 0; rank < kNumProcs; rank++) {
    MPI_Comm comm(rank, kNumProcs);
    double send_data[kArraySize];
    double recv_data[kArraySize];

    for (int j = 0; j < kArraySize; j++) {
      send_data[j] = send_values[rank][j];
      recv_data[j] = 0;
    }

    my_allreduce(send_data, recv_data, kArraySize, MPI_DOUBLE, MPI_MIN, &comm);

    for (int j = 0; j < kArraySize; j++) {
      recv_values[rank][j] = recv_data[j];
    }
  }

  std::vector<double> expected(kArraySize, 1e9);
  for (int i = 0; i < kNumProcs; i++) {
    for (int j = 0; j < kArraySize; j++) {
      if (send_values[i][j] < expected[j]) {
        expected[j] = send_values[i][j];
      }
    }
  }

  for (int rank = 0; rank < kNumProcs; rank++) {
    for (int j = 0; j < kArraySize; j++) {
      EXPECT_EQ(recv_values[rank][j], expected[j]);
    }
  }
}
