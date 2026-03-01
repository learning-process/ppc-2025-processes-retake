#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "kaur_a_vert_ribbon_scheme/common/include/common.hpp"
#include "kaur_a_vert_ribbon_scheme/mpi/include/ops_mpi.hpp"
#include "kaur_a_vert_ribbon_scheme/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kaur_a_vert_ribbon_scheme {

class KaurAVertRibbonSchemeFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);
    rows_ = size;
    cols_ = size;

    matrix_.resize(static_cast<std::size_t>(rows_) * cols_);
    vector_.resize(cols_);
    expected_.resize(rows_, 0.0);

    for (int j = 0; j < cols_; j++) {
      vector_[j] = static_cast<double>((j % 10) + 1);
      for (int i = 0; i < rows_; i++) {
        matrix_[static_cast<std::size_t>(j * rows_) + i] = static_cast<double>(((i + j) % 20) - 10);
      }
    }

    for (int j = 0; j < cols_; j++) {
      for (int i = 0; i < rows_; i++) {
        expected_[i] += matrix_[static_cast<std::size_t>(j * rows_) + i] * vector_[j];
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_.size()) {
      return false;
    }
    for (std::size_t i = 0; i < expected_.size(); i++) {
      if (std::abs(output_data[i] - expected_[i]) > 1e-9) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    TaskData data;
    data.matrix = matrix_;
    data.vector = vector_;
    data.rows = rows_;
    data.cols = cols_;
    return data;
  }

 private:
  std::vector<double> matrix_;
  std::vector<double> vector_;
  std::vector<double> expected_;
  int rows_ = 0;
  int cols_ = 0;
};

namespace {

TEST_P(KaurAVertRibbonSchemeFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KaurAVertRibbonSchemeMPI, InType>(kTestParam, PPC_SETTINGS_kaur_a_vert_ribbon_scheme),
    ppc::util::AddFuncTask<KaurAVertRibbonSchemeSEQ, InType>(kTestParam, PPC_SETTINGS_kaur_a_vert_ribbon_scheme));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KaurAVertRibbonSchemeFuncTests::PrintFuncTestName<KaurAVertRibbonSchemeFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KaurAVertRibbonSchemeFuncTests, kGtestValues, kPerfTestName);

TEST(KaurAVertRibbonSchemeTest, SingleElementMatrix) {
  TaskData data;
  data.rows = 1;
  data.cols = 1;
  data.matrix = {5.0};
  data.vector = {3.0};

  KaurAVertRibbonSchemeSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto &output = task.GetOutput();
  ASSERT_EQ(output.size(), 1);
  ASSERT_NEAR(output[0], 15.0, 1e-9);
}

TEST(KaurAVertRibbonSchemeTest, IdentityMatrix) {
  TaskData data;
  data.rows = 3;
  data.cols = 3;
  data.matrix = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  data.vector = {2.0, 4.0, 6.0};

  KaurAVertRibbonSchemeSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto &output = task.GetOutput();
  ASSERT_EQ(output.size(), 3);
  ASSERT_NEAR(output[0], 2.0, 1e-9);
  ASSERT_NEAR(output[1], 4.0, 1e-9);
  ASSERT_NEAR(output[2], 6.0, 1e-9);
}

TEST(KaurAVertRibbonSchemeTest, ZeroMatrix) {
  TaskData data;
  data.rows = 2;
  data.cols = 2;
  data.matrix = {0.0, 0.0, 0.0, 0.0};
  data.vector = {1.0, 1.0};

  KaurAVertRibbonSchemeSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto &output = task.GetOutput();
  ASSERT_EQ(output.size(), 2);
  ASSERT_NEAR(output[0], 0.0, 1e-9);
  ASSERT_NEAR(output[1], 0.0, 1e-9);
}

TEST(KaurAVertRibbonSchemeTest, ZeroVector) {
  TaskData data;
  data.rows = 2;
  data.cols = 2;
  data.matrix = {1.0, 2.0, 3.0, 4.0};
  data.vector = {0.0, 0.0};

  KaurAVertRibbonSchemeSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto &output = task.GetOutput();
  ASSERT_EQ(output.size(), 2);
  ASSERT_NEAR(output[0], 0.0, 1e-9);
  ASSERT_NEAR(output[1], 0.0, 1e-9);
}

TEST(KaurAVertRibbonSchemeTest, RectangularMatrixMoreRows) {
  TaskData data;
  data.rows = 4;
  data.cols = 2;
  data.matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  data.vector = {1.0, 2.0};

  KaurAVertRibbonSchemeSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto &output = task.GetOutput();
  ASSERT_EQ(output.size(), 4);
  ASSERT_NEAR(output[0], 11.0, 1e-9);
  ASSERT_NEAR(output[1], 14.0, 1e-9);
  ASSERT_NEAR(output[2], 17.0, 1e-9);
  ASSERT_NEAR(output[3], 20.0, 1e-9);
}

TEST(KaurAVertRibbonSchemeTest, RectangularMatrixMoreCols) {
  TaskData data;
  data.rows = 2;
  data.cols = 4;
  data.matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  data.vector = {1.0, 1.0, 1.0, 1.0};

  KaurAVertRibbonSchemeSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto &output = task.GetOutput();
  ASSERT_EQ(output.size(), 2);
  ASSERT_NEAR(output[0], 16.0, 1e-9);
  ASSERT_NEAR(output[1], 20.0, 1e-9);
}

TEST(KaurAVertRibbonSchemeTest, NegativeValues) {
  TaskData data;
  data.rows = 2;
  data.cols = 2;
  data.matrix = {-1.0, -2.0, -3.0, -4.0};
  data.vector = {-1.0, -2.0};

  KaurAVertRibbonSchemeSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto &output = task.GetOutput();
  ASSERT_EQ(output.size(), 2);
  ASSERT_NEAR(output[0], 7.0, 1e-9);
  ASSERT_NEAR(output[1], 10.0, 1e-9);
}

TEST(KaurAVertRibbonSchemeTest, SingleRow) {
  TaskData data;
  data.rows = 1;
  data.cols = 5;
  data.matrix = {1.0, 2.0, 3.0, 4.0, 5.0};
  data.vector = {1.0, 2.0, 3.0, 4.0, 5.0};

  KaurAVertRibbonSchemeSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto &output = task.GetOutput();
  ASSERT_EQ(output.size(), 1);
  ASSERT_NEAR(output[0], 55.0, 1e-9);
}

TEST(KaurAVertRibbonSchemeTest, SingleColumn) {
  TaskData data;
  data.rows = 5;
  data.cols = 1;
  data.matrix = {1.0, 2.0, 3.0, 4.0, 5.0};
  data.vector = {2.0};

  KaurAVertRibbonSchemeSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto &output = task.GetOutput();
  ASSERT_EQ(output.size(), 5);
  ASSERT_NEAR(output[0], 2.0, 1e-9);
  ASSERT_NEAR(output[1], 4.0, 1e-9);
  ASSERT_NEAR(output[2], 6.0, 1e-9);
  ASSERT_NEAR(output[3], 8.0, 1e-9);
  ASSERT_NEAR(output[4], 10.0, 1e-9);
}

TEST(KaurAVertRibbonSchemeTest, FloatingPointPrecision) {
  TaskData data;
  data.rows = 2;
  data.cols = 2;
  data.matrix = {0.1, 0.2, 0.3, 0.4};
  data.vector = {0.5, 0.5};

  KaurAVertRibbonSchemeSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto &output = task.GetOutput();
  ASSERT_EQ(output.size(), 2);
  ASSERT_NEAR(output[0], 0.2, 1e-9);
  ASSERT_NEAR(output[1], 0.3, 1e-9);
}

TEST(KaurAVertRibbonSchemeValidationTest, InvalidRowsZero) {
  TaskData data;
  data.rows = 0;
  data.cols = 2;
  data.matrix = {};
  data.vector = {1.0, 2.0};

  KaurAVertRibbonSchemeSEQ task(data);
  EXPECT_FALSE(task.Validation());
}

TEST(KaurAVertRibbonSchemeValidationTest, InvalidColsZero) {
  TaskData data;
  data.rows = 2;
  data.cols = 0;
  data.matrix = {};
  data.vector = {};

  KaurAVertRibbonSchemeSEQ task(data);
  EXPECT_FALSE(task.Validation());
}

TEST(KaurAVertRibbonSchemeValidationTest, InvalidMatrixSize) {
  TaskData data;
  data.rows = 2;
  data.cols = 2;
  data.matrix = {1.0, 2.0, 3.0};
  data.vector = {1.0, 2.0};

  KaurAVertRibbonSchemeSEQ task(data);
  EXPECT_FALSE(task.Validation());
}

TEST(KaurAVertRibbonSchemeValidationTest, InvalidVectorSize) {
  TaskData data;
  data.rows = 2;
  data.cols = 2;
  data.matrix = {1.0, 2.0, 3.0, 4.0};
  data.vector = {1.0};

  KaurAVertRibbonSchemeSEQ task(data);
  EXPECT_FALSE(task.Validation());
}

}  // namespace
}  // namespace kaur_a_vert_ribbon_scheme
