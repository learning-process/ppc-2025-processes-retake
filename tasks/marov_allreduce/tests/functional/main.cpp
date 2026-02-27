#include <gtest/gtest.h>
#include <array>
#include <string>
#include <vector>

#include "marov_allreduce/mpi/include/ops_mpi.h"

TEST(marovAllreduceMpi, sumInts) {
  const int k_num_procs = 7;
  std::vector<int> send_values(k_num_procs);
  std::vector<int> recv_values(k_num_procs);

  for (int i = 0; i < k_num_procs; i++) {
    send_values[i] = i + 1;
  }

  for (int rank = 0; rank < k_num_procs; rank++) {
    MPI_Comm comm(rank, k_num_procs);
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
  const int k_num_procs = 7;
  std::vector<float> send_values(k_num_procs);
  std::vector<float> recv_values(k_num_procs);

  for (int i = 0; i < k_num_procs; i++) {
    send_values[i] = static_cast<float>(i) * 1.5F;
  }

  for (int rank = 0; rank < k_num_procs; rank++) {
    MPI_Comm comm(rank, k_num_procs);
    float send_data = send_values[rank];
    float recv_data = 0;

    my_allreduce(&send_data, &recv_data, 1, MPI_FLOAT, MPI_MAX, &comm);
    recv_values[rank] = recv_data;
  }

  float expected = 0;
  for (float v : send_values) {
    expected = std::max(v, expected);
  }
  for (float r : recv_values) {
    EXPECT_EQ(r, expected);
  }
}

TEST(marovAllreduceMpi, minDoubles) {
  const int k_num_procs = 7;
  const int k_array_size = 3;
  std::vector<std::vector<double>> send_values(
      k_num_procs, std::vector<double>(k_array_size));
  std::vector<std::vector<double>> recv_values(
      k_num_procs, std::vector<double>(k_array_size));

  for (int i = 0; i < k_num_procs; i++) {
    send_values[i][0] = i + 1.0;
    send_values[i][1] = i * 2.0;
    send_values[i][2] = i + 0.5;
  }

  for (int rank = 0; rank < k_num_procs; rank++) {
    MPI_Comm comm(rank, k_num_procs);
    std::array<double, k_array_size> send_data{};
    std::array<double, k_array_size> recv_data{};

    for (int j = 0; j < k_array_size; j++) {
      send_data[j] = send_values[rank][j];
      recv_data[j] = 0;
    }

    my_allreduce(send_data.data(), recv_data.data(), k_array_size, MPI_DOUBLE,
                 MPI_MIN, &comm);

    for (int j = 0; j < k_array_size; j++) {
      recv_values[rank][j] = recv_data[j];
    }
  }

  std::vector<double> expected(k_array_size, 1e9);
  for (int i = 0; i < k_num_procs; i++) {
    for (int j = 0; j < k_array_size; j++) {
      expected[j] = std::min(send_values[i][j], expected[j]);
    }
  }

  for (int rank = 0; rank < k_num_procs; rank++) {
    for (int j = 0; j < k_array_size; j++) {
      EXPECT_EQ(recv_values[rank][j], expected[j]);
    }
  }
}
