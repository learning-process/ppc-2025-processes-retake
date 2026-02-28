#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "util/include/util.hpp"
#include "yushkova_p_radix_sort_with_simple_merge/mpi/include/ops_mpi.hpp"
#include "yushkova_p_radix_sort_with_simple_merge/seq/include/ops_seq.hpp"

namespace yushkova_p_radix_sort_with_simple_merge {
namespace {

bool AlmostEqual(double a, double b) {
  return std::abs(a - b) < 1e-12;
}

bool SameVectors(const std::vector<double> &a, const std::vector<double> &b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!AlmostEqual(a[i], b[i])) {
      return false;
    }
  }
  return true;
}

std::vector<double> BuildReference(const std::vector<double> &input) {
  std::vector<double> expected = input;
  std::sort(expected.begin(), expected.end());
  return expected;
}

}  // namespace

TEST(YushkovaRadixSortSeq, SortsMixedValues) {
  const std::vector<double> input = {2.5, -3.1, 0.0, 8.2, -1.0, 2.5, 7.7, -9.9};
  const std::vector<double> expected = BuildReference(input);

  YushkovaPRadixSortWithSimpleMergeSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_TRUE(SameVectors(std::get<0>(task.GetOutput()), expected));
  EXPECT_EQ(std::get<1>(task.GetOutput()), 0);
}

TEST(YushkovaRadixSortSeq, HandlesEdgeCases) {
  {
    const std::vector<double> empty_input;
    YushkovaPRadixSortWithSimpleMergeSEQ task(empty_input);
    ASSERT_TRUE(task.Validation());
    ASSERT_TRUE(task.PreProcessing());
    ASSERT_TRUE(task.Run());
    ASSERT_TRUE(task.PostProcessing());
    EXPECT_TRUE(std::get<0>(task.GetOutput()).empty());
  }

  {
    const std::vector<double> one_element = {42.0};
    YushkovaPRadixSortWithSimpleMergeSEQ task(one_element);
    ASSERT_TRUE(task.Validation());
    ASSERT_TRUE(task.PreProcessing());
    ASSERT_TRUE(task.Run());
    ASSERT_TRUE(task.PostProcessing());
    EXPECT_TRUE(SameVectors(std::get<0>(task.GetOutput()), one_element));
  }
}

TEST(YushkovaRadixSortMpi, MatchesStdSort) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const std::vector<double> input = {9.9, -2.3, -2.3, 7.1, 0.0, 6.6, -8.4, 5.5, 1.2, -1.0, 3.3, 3.2};
  const std::vector<double> expected = BuildReference(input);

  YushkovaPRadixSortWithSimpleMergeMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    EXPECT_TRUE(SameVectors(std::get<0>(task.GetOutput()), expected));
  }
}

TEST(YushkovaRadixSortMpi, WorksWithNegativesAndFractions) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const std::vector<double> input = {-0.5, -100.125, 0.25, 0.5, -0.25, 3.1415, -7.75, 2.7182, 0.0};
  const std::vector<double> expected = BuildReference(input);

  YushkovaPRadixSortWithSimpleMergeMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    EXPECT_TRUE(SameVectors(std::get<0>(task.GetOutput()), expected));
  }
}

}  // namespace yushkova_p_radix_sort_with_simple_merge
