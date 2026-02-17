#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
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

}  // namespace

class YushkovaPMinInMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType& test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto test_param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(test_param);
  }

  bool CheckTestOutputData(OutType& output_data) final {
    if (output_data.size() != static_cast<size_t>(input_data_)) {
      return false;
    }

    for (InType i = 0; i < input_data_; ++i) {
      InType expected = GenerateValue(static_cast<int64_t>(i), 0);
      for (InType j = 1; j < input_data_; j++) {
        expected = std::min(expected, GenerateValue(static_cast<int64_t>(i), static_cast<int64_t>(j)));
      }
      if (output_data[static_cast<size_t>(i)] != expected) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

TEST_P(YushkovaPMinInMatrixFuncTests, TestMinInMatrix) {
  ExecuteTest(GetParam());
}

namespace {

const std::array<TestType, 6> kTestParam = {std::make_tuple(1, "size_1"),     std::make_tuple(5, "size_5"),
                                            std::make_tuple(17, "size_17"),   std::make_tuple(64, "size_64"),
                                            std::make_tuple(128, "size_128"), std::make_tuple(256, "size_256")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<YushkovaPMinInMatrixMPI, InType>(kTestParam, PPC_SETTINGS_yushkova_p_min_in_matrix),
    ppc::util::AddFuncTask<YushkovaPMinInMatrixSEQ, InType>(kTestParam, PPC_SETTINGS_yushkova_p_min_in_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = YushkovaPMinInMatrixFuncTests::PrintFuncTestName<YushkovaPMinInMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(SequentialAndMPI, YushkovaPMinInMatrixFuncTests, kGtestValues, kPerfTestName);

}  // namespace

template <typename TaskType>
void ExpectPipelineSuccess(InType n) {
  auto task = std::make_shared<TaskType>(n);

  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());

  ASSERT_EQ(task->GetOutput().size(), static_cast<size_t>(n));
}

TEST(YushkovaMinMatrixStandalone, SeqPipelineWorks) {
  const std::array<InType, 6> sizes = {1, 4, 17, 33, 127, 255};
  for (auto s : sizes) {
    ExpectPipelineSuccess<YushkovaPMinInMatrixSEQ>(s);
  }
}

TEST(YushkovaMinMatrixStandalone, MpiPipelineWorks) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  const std::array<InType, 6> sizes = {1, 5, 18, 37, 130, 257};
  for (auto s : sizes) {
    ExpectPipelineSuccess<YushkovaPMinInMatrixMPI>(s);
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

template <typename TaskType>
void RunTwice(TaskType& task, InType n) {
  task.GetInput() = n;
  task.GetOutput().clear();

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  ASSERT_EQ(task.GetOutput().size(), static_cast<size_t>(n));
}

TEST(YushkovaMinMatrixPipeline, SeqReusable) {
  YushkovaPMinInMatrixSEQ task(4);
  RunTwice(task, 4);
  RunTwice(task, 9);
}

TEST(YushkovaMinMatrixPipeline, MpiReusable) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  YushkovaPMinInMatrixMPI task(6);
  RunTwice(task, 6);
  RunTwice(task, 14);
}

}  // namespace yushkova_p_min_in_matrix
