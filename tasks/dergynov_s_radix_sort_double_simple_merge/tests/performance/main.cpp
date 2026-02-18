#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include "dergynov_s_radix_sort_double_simple_merge/mpi/include/ops_mpi.hpp"
#include "dergynov_s_radix_sort_double_simple_merge/seq/include/ops_seq.hpp"

namespace dergynov_s_radix_sort_double_simple_merge {
namespace {

template <typename T>
void ValidateTask(T &task) {
  ASSERT_TRUE(task.Validation());
}

template <typename T>
void ProcessTask(T &task) {
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

template <typename T>
void ValidateAndProcess(T &task) {
  ValidateTask(task);
  ProcessTask(task);
}

template <typename T>
void CheckSortedAndSize(T &task, const std::vector<double> &data) {
  auto sorted = std::get<0>(task.GetOutput());
  EXPECT_TRUE(std::is_sorted(sorted.begin(), sorted.end()));
  EXPECT_EQ(sorted.size(), data.size());
}

class DergynovRadixSortPerfTests : public ::testing::Test {
 protected:
  void SetUp() override {
    const size_t k_data_size = 1000000;
    data.resize(k_data_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (size_t i = 0; i < k_data_size; ++i) {
      data[i] = dist(gen);
    }
  }

  std::vector<double> data;
};

TEST_F(DergynovRadixSortPerfTests, SeqPerformance) {
  DergynovSRadixSortDoubleSimpleMergeSEQ task(data);
  ValidateAndProcess(task);
  CheckSortedAndSize(task, data);
}

TEST_F(DergynovRadixSortPerfTests, MpiPerformance) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  DergynovSRadixSortDoubleSimpleMergeMPI task(data);
  ValidateAndProcess(task);

  if (rank == 0) {
    CheckSortedAndSize(task, data);
  }
}

}  // namespace
}  // namespace dergynov_s_radix_sort_double_simple_merge
