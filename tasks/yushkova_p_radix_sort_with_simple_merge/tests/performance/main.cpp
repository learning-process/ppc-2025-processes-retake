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
bool RunTask(TaskType &task) {
  return task.Validation() && task.PreProcessing() && task.Run() && task.PostProcessing();
}

std::vector<double> MakeRandomInput(std::size_t size) {
  std::mt19937_64 gen(42);
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
  ASSERT_TRUE(RunTask(task));

  const auto &output = std::get<0>(task.GetOutput());
  EXPECT_TRUE(std::is_sorted(output.begin(), output.end()));
  EXPECT_EQ(output.size(), data.size());
}

TEST_F(YushkovaRadixSortPerf, MpiPerformanceRun) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  YushkovaPRadixSortWithSimpleMergeMPI task(data);
  ASSERT_TRUE(RunTask(task));

  if (rank == 0) {
    const auto &output = std::get<0>(task.GetOutput());
    EXPECT_TRUE(std::is_sorted(output.begin(), output.end()));
    EXPECT_EQ(output.size(), data.size());
  }
}

}  // namespace yushkova_p_radix_sort_with_simple_merge
