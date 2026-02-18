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

class DergynovRadixSortPerfTests : public ::testing::Test {
 protected:
  void SetUp() override {
    const size_t kDataSize = 1000000;
    data_.resize(kDataSize);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (size_t i = 0; i < kDataSize; ++i) {
      data_[i] = dist(gen);
    }
  }

  std::vector<double> data_;
};

template <typename T>
void ValidateAndProcess(T &task) {
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

template <typename T>
void CheckSortedAndSize(T &task, const std::vector<double> &data) {
  auto sorted = std::get<0>(task.GetOutput());
  EXPECT_TRUE(std::is_sorted(sorted.begin(), sorted.end()));
  EXPECT_EQ(sorted.size(), data.size());
}

TEST_F(DergynovRadixSortPerfTests, SeqPerformance) {
  DergynovSRadixSortDoubleSimpleMergeSEQ task(data_);
  ValidateAndProcess(task);
  CheckSortedAndSize(task, data_);
}

TEST_F(DergynovRadixSortPerfTests, MpiPerformance) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  DergynovSRadixSortDoubleSimpleMergeMPI task(data_);
  ValidateAndProcess(task);

  if (rank == 0) {
    CheckSortedAndSize(task, data_);
  }
}

}  // namespace
}  // namespace dergynov_s_radix_sort_double_simple_merge
