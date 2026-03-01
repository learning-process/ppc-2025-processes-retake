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

// Структура для успешных тестов
struct ValidTestCase {
  std::string name;
  TaskData data;
  std::vector<double> expected;
};

class KaurAVertRibbonSchemeValidTest : public ::testing::TestWithParam<ValidTestCase> {};

TEST_P(KaurAVertRibbonSchemeValidTest, CheckValidCase) {
  const auto &param = GetParam();
  KaurAVertRibbonSchemeSEQ task(param.data);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto &output = task.GetOutput();
  ASSERT_EQ(output.size(), param.expected.size());
  for (std::size_t i = 0; i < param.expected.size(); i++) {
    ASSERT_NEAR(output[i], param.expected[i], 1e-9);
  }
}

INSTANTIATE_TEST_SUITE_P(
    KaurAVertRibbonSchemeValidTests, KaurAVertRibbonSchemeValidTest,
    ::testing::Values(
        ValidTestCase{"SingleElementMatrix", TaskData{.matrix = {5.0}, .vector = {3.0}, .rows = 1, .cols = 1}, {15.0}},
        ValidTestCase{"IdentityMatrix",
                      TaskData{.matrix = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
                               .vector = {2.0, 4.0, 6.0},
                               .rows = 3,
                               .cols = 3},
                      {2.0, 4.0, 6.0}},
        ValidTestCase{"ZeroMatrix",
                      TaskData{.matrix = {0.0, 0.0, 0.0, 0.0}, .vector = {1.0, 1.0}, .rows = 2, .cols = 2},
                      {0.0, 0.0}},
        ValidTestCase{"ZeroVector",
                      TaskData{.matrix = {1.0, 2.0, 3.0, 4.0}, .vector = {0.0, 0.0}, .rows = 2, .cols = 2},
                      {0.0, 0.0}},
        ValidTestCase{
            "RectangularMatrixMoreRows",
            TaskData{.matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, .vector = {1.0, 2.0}, .rows = 4, .cols = 2},
            {11.0, 14.0, 17.0, 20.0}},
        ValidTestCase{"RectangularMatrixMoreCols",
                      TaskData{.matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                               .vector = {1.0, 1.0, 1.0, 1.0},
                               .rows = 2,
                               .cols = 4},
                      {16.0, 20.0}},
        ValidTestCase{"NegativeValues",
                      TaskData{.matrix = {-1.0, -2.0, -3.0, -4.0}, .vector = {-1.0, -2.0}, .rows = 2, .cols = 2},
                      {7.0, 10.0}},
        ValidTestCase{
            "SingleRow",
            TaskData{.matrix = {1.0, 2.0, 3.0, 4.0, 5.0}, .vector = {1.0, 2.0, 3.0, 4.0, 5.0}, .rows = 1, .cols = 5},
            {55.0}},
        ValidTestCase{"SingleColumn",
                      TaskData{.matrix = {1.0, 2.0, 3.0, 4.0, 5.0}, .vector = {2.0}, .rows = 5, .cols = 1},
                      {2.0, 4.0, 6.0, 8.0, 10.0}},
        ValidTestCase{"FloatingPointPrecision",
                      TaskData{.matrix = {0.1, 0.2, 0.3, 0.4}, .vector = {0.5, 0.5}, .rows = 2, .cols = 2},
                      {0.2, 0.3}}),
    [](const ::testing::TestParamInfo<ValidTestCase> &info) { return info.param.name; });

struct InvalidTestCase {
  std::string name;
  TaskData data;
};

class KaurAVertRibbonSchemeInvalidTest : public ::testing::TestWithParam<InvalidTestCase> {};

TEST_P(KaurAVertRibbonSchemeInvalidTest, CheckInvalidCase) {
  const auto &param = GetParam();
  KaurAVertRibbonSchemeSEQ task(param.data);
  EXPECT_FALSE(task.Validation());
}

INSTANTIATE_TEST_SUITE_P(
    KaurAVertRibbonSchemeInvalidTests, KaurAVertRibbonSchemeInvalidTest,
    ::testing::Values(InvalidTestCase{"InvalidRowsZero",
                                      TaskData{.matrix = {}, .vector = {1.0, 2.0}, .rows = 0, .cols = 2}},
                      InvalidTestCase{"InvalidColsZero", TaskData{.matrix = {}, .vector = {}, .rows = 2, .cols = 0}},
                      InvalidTestCase{"InvalidMatrixSize",
                                      TaskData{.matrix = {1.0, 2.0, 3.0}, .vector = {1.0, 2.0}, .rows = 2, .cols = 2}},
                      InvalidTestCase{"InvalidVectorSize",
                                      TaskData{.matrix = {1.0, 2.0, 3.0, 4.0}, .vector = {1.0}, .rows = 2, .cols = 2}}),
    [](const ::testing::TestParamInfo<InvalidTestCase> &info) { return info.param.name; });

}  // namespace
}  // namespace kaur_a_vert_ribbon_scheme
