#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "sokolov_k_min_val_matrix/common/include/common.hpp"
#include "sokolov_k_min_val_matrix/mpi/include/ops_mpi.hpp"
#include "sokolov_k_min_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace k_sokolov_min_val_matrix {

class SokolovKMinValMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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
    return output_data == 1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(SokolovKMinValMatrixFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {
    std::make_tuple(1, "1"), std::make_tuple(2, "2"),   std::make_tuple(3, "3"),  std::make_tuple(5, "5"),
    std::make_tuple(7, "7"), std::make_tuple(10, "10"), std::make_tuple(50, "50")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SokolovKMinValMatrixMPI, InType>(kTestParam, PPC_SETTINGS_sokolov_k_min_val_matrix),
    ppc::util::AddFuncTask<SokolovKMinValMatrixSEQ, InType>(kTestParam, PPC_SETTINGS_sokolov_k_min_val_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SokolovKMinValMatrixFuncTests::PrintFuncTestName<SokolovKMinValMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, SokolovKMinValMatrixFuncTests, kGtestValues, kPerfTestName);

TEST(SokolovKMinValMatrixSeqValidation, RejectsZeroInput) {
  SokolovKMinValMatrixSEQ task(0);
  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(SokolovKMinValMatrixSeqValidation, RejectsNegativeInput) {
  SokolovKMinValMatrixSEQ task(-5);
  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(SokolovKMinValMatrixSeqTest, SingleElementMatrix) {
  SokolovKMinValMatrixSEQ task(1);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);
}

TEST(SokolovKMinValMatrixSeqTest, SmallMatrix2x2) {
  SokolovKMinValMatrixSEQ task(2);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);
}

TEST(SokolovKMinValMatrixSeqTest, MediumMatrix100x100) {
  SokolovKMinValMatrixSEQ task(100);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);
}

TEST(SokolovKMinValMatrixSeqTest, LargeMatrix200x200) {
  SokolovKMinValMatrixSEQ task(200);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);
}

TEST(SokolovKMinValMatrixMpiValidation, RejectsZeroInput) {
  SokolovKMinValMatrixMPI task(0);
  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(SokolovKMinValMatrixMpiValidation, RejectsNegativeInput) {
  SokolovKMinValMatrixMPI task(-3);
  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(SokolovKMinValMatrixMpiTest, SingleElementMatrix) {
  SokolovKMinValMatrixMPI task(1);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);
}

TEST(SokolovKMinValMatrixMpiTest, SmallMatrix3x3) {
  SokolovKMinValMatrixMPI task(3);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);
}

TEST(SokolovKMinValMatrixMpiTest, MediumMatrix100x100) {
  SokolovKMinValMatrixMPI task(100);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);
}

}  // namespace

}  // namespace k_sokolov_min_val_matrix
