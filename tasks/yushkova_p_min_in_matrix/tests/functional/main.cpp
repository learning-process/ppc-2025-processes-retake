#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "util/include/util.hpp"
#include "yushkova_p_min_in_matrix/common/include/common.hpp"
#include "yushkova_p_min_in_matrix/mpi/include/ops_mpi.hpp"
#include "yushkova_p_min_in_matrix/seq/include/ops_seq.hpp"

namespace yushkova_p_min_in_matrix {

namespace {

InType GenerateValue(int64_t i, int64_t j) {
  constexpr int64_t kA = 1103515245LL;
  constexpr int64_t kC = 12345LL;
  constexpr int64_t kM = 2147483648LL;

  int64_t seed = ((i % kM) * (100000007LL % kM) + (j % kM) * (1000000009LL % kM)) % kM;
  seed = (seed ^ 42LL) % kM;
  int64_t val = ((kA % kM) * (seed % kM) + kC) % kM;
  return static_cast<InType>((val % 2000001LL) - 1000000LL);
}

OutType BuildExpectedOutput(InType n) {
  OutType expected;
  expected.reserve(static_cast<size_t>(n));
  for (InType i = 0; i < n; ++i) {
    InType row_min = GenerateValue(static_cast<int64_t>(i), 0);
    for (InType j = 1; j < n; ++j) {
      row_min = std::min(row_min, GenerateValue(static_cast<int64_t>(i), static_cast<int64_t>(j)));
    }
    expected.push_back(row_min);
  }
  return expected;
}

template <typename TaskType>
bool RunPipeline(TaskType &task) {
  return task.Validation() && task.PreProcessing() && task.Run() && task.PostProcessing();
}

template <typename TaskType>
OutType RunPipelineForN(InType n) {
  auto task = std::make_shared<TaskType>(n);
  if (!RunPipeline(*task)) {
    return {};
  }
  return task->GetOutput();
}

}  // namespace

TEST(YushkovaMinMatrixFunctional, SeqSizes) {
  constexpr std::array<InType, 6> kSizes = {1, 5, 17, 64, 128, 256};
  for (InType n : kSizes) {
    const OutType expected = BuildExpectedOutput(n);
    const OutType actual = RunPipelineForN<YushkovaPMinInMatrixSEQ>(n);
    ASSERT_EQ(actual.size(), static_cast<size_t>(n));
    EXPECT_EQ(actual, expected);
  }
}

TEST(YushkovaMinMatrixFunctional, MpiSizes) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  constexpr std::array<InType, 6> kSizes = {1, 5, 17, 64, 128, 256};
  for (InType n : kSizes) {
    const OutType expected = BuildExpectedOutput(n);
    const OutType actual = RunPipelineForN<YushkovaPMinInMatrixMPI>(n);
    ASSERT_EQ(actual.size(), static_cast<size_t>(n));
    EXPECT_EQ(actual, expected);
  }
}

TEST(YushkovaMinMatrixValidation, RejectsZeroSeq) {
  {
    YushkovaPMinInMatrixSEQ task(0);
    EXPECT_FALSE(task.Validation());
  }
  ppc::util::DestructorFailureFlag::Unset();
}

TEST(YushkovaMinMatrixValidation, RejectsZeroMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  {
    YushkovaPMinInMatrixMPI task(0);
    EXPECT_FALSE(task.Validation());
  }
  ppc::util::DestructorFailureFlag::Unset();
}

TEST(YushkovaMinMatrixPipeline, SeqReusable) {
  YushkovaPMinInMatrixSEQ task(4);

  constexpr std::array<InType, 2> kSizes = {4, 9};
  for (InType n : kSizes) {
    task.GetInput() = n;
    task.GetOutput().clear();
    ASSERT_TRUE(RunPipeline(task));
    EXPECT_EQ(task.GetOutput().size(), static_cast<size_t>(n));
  }
}

TEST(YushkovaMinMatrixPipeline, MpiReusable) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  YushkovaPMinInMatrixMPI task(6);

  constexpr std::array<InType, 2> kSizes = {6, 14};
  for (InType n : kSizes) {
    task.GetInput() = n;
    task.GetOutput().clear();
    ASSERT_TRUE(RunPipeline(task));
    EXPECT_EQ(task.GetOutput().size(), static_cast<size_t>(n));
  }
}

}  // namespace yushkova_p_min_in_matrix
