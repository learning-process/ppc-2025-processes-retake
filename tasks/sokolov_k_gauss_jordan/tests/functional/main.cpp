#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>

#include "sokolov_k_gauss_jordan/common/include/common.hpp"
#include "sokolov_k_gauss_jordan/mpi/include/ops_mpi.hpp"
#include "sokolov_k_gauss_jordan/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sokolov_k_gauss_jordan {

class SokolovKGaussJordanFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(SokolovKGaussJordanFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SokolovKGaussJordanMPI, InType>(kTestParam, PPC_SETTINGS_sokolov_k_gauss_jordan),
    ppc::util::AddFuncTask<SokolovKGaussJordanSEQ, InType>(kTestParam, PPC_SETTINGS_sokolov_k_gauss_jordan));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SokolovKGaussJordanFuncTests::PrintFuncTestName<SokolovKGaussJordanFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, SokolovKGaussJordanFuncTests, kGtestValues, kPerfTestName);

TEST(SokolovKGaussJordanSEQFunc, SolveSize1) {
  auto task = std::make_shared<SokolovKGaussJordanSEQ>(1);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 1);
}

TEST(SokolovKGaussJordanSEQFunc, SolveSize2) {
  auto task = std::make_shared<SokolovKGaussJordanSEQ>(2);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 2);
}

TEST(SokolovKGaussJordanSEQFunc, SolveSize4) {
  auto task = std::make_shared<SokolovKGaussJordanSEQ>(4);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 4);
}

TEST(SokolovKGaussJordanSEQFunc, SolveSize10) {
  auto task = std::make_shared<SokolovKGaussJordanSEQ>(10);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 10);
}

TEST(SokolovKGaussJordanSEQFunc, SolveSize50) {
  auto task = std::make_shared<SokolovKGaussJordanSEQ>(50);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 50);
}

TEST(SokolovKGaussJordanSEQFunc, SolveSize100) {
  auto task = std::make_shared<SokolovKGaussJordanSEQ>(100);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 100);
}

TEST(SokolovKGaussJordanSEQFunc, ValidationFailsZero) {
  auto task = std::make_shared<SokolovKGaussJordanSEQ>(0);
  EXPECT_FALSE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
}

TEST(SokolovKGaussJordanSEQFunc, ValidationFailsNegative) {
  auto task = std::make_shared<SokolovKGaussJordanSEQ>(-5);
  EXPECT_FALSE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
}

TEST(SokolovKGaussJordanMPIFunc, SolveSize1) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  auto task = std::make_shared<SokolovKGaussJordanMPI>(1);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 1);
}

TEST(SokolovKGaussJordanMPIFunc, SolveSize2) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  auto task = std::make_shared<SokolovKGaussJordanMPI>(2);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 2);
}

TEST(SokolovKGaussJordanMPIFunc, SolveSize4) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  auto task = std::make_shared<SokolovKGaussJordanMPI>(4);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 4);
}

TEST(SokolovKGaussJordanMPIFunc, SolveSize10) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  auto task = std::make_shared<SokolovKGaussJordanMPI>(10);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 10);
}

TEST(SokolovKGaussJordanMPIFunc, SolveSize50) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  auto task = std::make_shared<SokolovKGaussJordanMPI>(50);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 50);
}

TEST(SokolovKGaussJordanMPIFunc, SolveSize100) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  auto task = std::make_shared<SokolovKGaussJordanMPI>(100);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 100);
}

TEST(SokolovKGaussJordanMPIFunc, ValidationFailsZero) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  auto task = std::make_shared<SokolovKGaussJordanMPI>(0);
  EXPECT_FALSE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
}

TEST(SokolovKGaussJordanMPIFunc, ValidationFailsNegative) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  auto task = std::make_shared<SokolovKGaussJordanMPI>(-5);
  EXPECT_FALSE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
}

}  // namespace

}  // namespace sokolov_k_gauss_jordan
