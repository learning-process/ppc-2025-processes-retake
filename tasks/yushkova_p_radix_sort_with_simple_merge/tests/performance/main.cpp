#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include "util/include/util.hpp"
#include "yushkova_p_radix_sort_with_simple_merge/mpi/include/ops_mpi.hpp"
#include "yushkova_p_radix_sort_with_simple_merge/seq/include/ops_seq.hpp"

namespace yushkova_p_radix_sort_with_simple_merge {
namespace {

template <class TaskType>
void ExecuteTask(TaskType &task) {
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

void CheckSortedOutput(const std::vector<double> &output, std::size_t size) {
  EXPECT_TRUE(std::ranges::is_sorted(output));
  EXPECT_EQ(output.size(), size);
}

std::vector<double> MakeRandomInput(std::size_t size) {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_real_distribution<double> dist(-10000.0, 10000.0);

  std::vector<double> data(size);
  for (double &value : data) {
    value = dist(gen);
  }
  return data;
}

}  // namespace

class YushkovaRadixSortPerf : public ::testing::Test {
 protected:
  void SetUp() override {
    data = MakeRandomInput(250000);
  }

  std::vector<double> data;
};

TEST_F(YushkovaRadixSortPerf, SeqPerformanceRun) {
  YushkovaPRadixSortWithSimpleMergeSEQ task(data);
  ExecuteTask(task);

  const auto &output = std::get<0>(task.GetOutput());
  CheckSortedOutput(output, data.size());
}

TEST_F(YushkovaRadixSortPerf, MpiPerformanceRun) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  YushkovaPRadixSortWithSimpleMergeMPI task(data);
  ExecuteTask(task);

  if (rank == 0) {
    const auto &output = std::get<0>(task.GetOutput());
    CheckSortedOutput(output, data.size());
  }
}

}  // namespace yushkova_p_radix_sort_with_simple_merge
