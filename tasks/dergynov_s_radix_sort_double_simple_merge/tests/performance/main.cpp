#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
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
    test_data_.resize(kDataSize);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (size_t i = 0; i < kDataSize; ++i) {
      test_data_[i] = dist(gen);
    }
  }

  std::vector<double> test_data_;
};

template <typename T>
void RunAndCheckTask(T &task, const std::vector<double> &test_data) {
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto sorted = std::get<0>(task.GetOutput());
  EXPECT_TRUE(std::is_sorted(sorted.begin(), sorted.end()));
  EXPECT_EQ(sorted.size(), test_data.size());
}

TEST_F(DergynovRadixSortPerfTests, SeqPerformance) {
  DergynovSRadixSortDoubleSimpleMergeSEQ task(test_data_);
  RunAndCheckTask(task, test_data_);
}

TEST_F(DergynovRadixSortPerfTests, MpiPerformance) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  DergynovSRadixSortDoubleSimpleMergeMPI task(test_data_);
  RunAndCheckTask(task, test_data_);

  if (rank == 0) {
    auto sorted = std::get<0>(task.GetOutput());
    EXPECT_TRUE(std::is_sorted(sorted.begin(), sorted.end()));
    EXPECT_EQ(sorted.size(), test_data_.size());
  }
}

}  // namespace
}  // namespace dergynov_s_radix_sort_double_simple_merge
