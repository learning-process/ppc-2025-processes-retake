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

static bool VectorsNear(const std::vector<double> &a, const std::vector<double> &b, double eps) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > eps) {
      return false;
    }
  }
  return true;
}

static bool RunAndCheck(const TaskData &data, const std::vector<double> &expected) {
  KaurAVertRibbonSchemeSEQ task(data);
  if (!task.Validation()) {
    return false;
  }
  if (!task.PreProcessing()) {
    return false;
  }
  if (!task.Run()) {
    return false;
  }
  if (!task.PostProcessing()) {
    return false;
  }
  return VectorsNear(task.GetOutput(), expected, 1e-9);
}

TEST(KaurAVertRibbonSchemeTest, SingleElementMatrix) {
  TaskData data;
  data.rows = 1;
  data.cols = 1;
  data.matrix = {5.0};
  data.vector = {3.0};
  std::vector<double> expected = {15.0};
  ASSERT_TRUE(RunAndCheck(data, expected));
}

TEST(KaurAVertRibbonSchemeTest, IdentityMatrix) {
  TaskData data;
  data.rows = 3;
  data.cols = 3;
  data.matrix = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  data.vector = {2.0, 4.0, 6.0};
  std::vector<double> expected = {2.0, 4.0, 6.0};
  ASSERT_TRUE(RunAndCheck(data, expected));
}

TEST(KaurAVertRibbonSchemeTest, ZeroMatrix) {
  TaskData data;
  data.rows = 2;
  data.cols = 2;
  data.matrix = {0.0, 0.0, 0.0, 0.0};
  data.vector = {1.0, 1.0};
  std::vector<double> expected = {0.0, 0.0};
  ASSERT_TRUE(RunAndCheck(data, expected));
}

TEST(KaurAVertRibbonSchemeTest, ZeroVector) {
  TaskData data;
  data.rows = 2;
  data.cols = 2;
  data.matrix = {1.0, 2.0, 3.0, 4.0};
  data.vector = {0.0, 0.0};
  std::vector<double> expected = {0.0, 0.0};
  ASSERT_TRUE(RunAndCheck(data, expected));
}

TEST(KaurAVertRibbonSchemeTest, RectangularMatrixMoreRows) {
  TaskData data;
  data.rows = 4;
  data.cols = 2;
  data.matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  data.vector = {1.0, 2.0};
  std::vector<double> expected = {11.0, 14.0, 17.0, 20.0};
  ASSERT_TRUE(RunAndCheck(data, expected));
}

TEST(KaurAVertRibbonSchemeTest, RectangularMatrixMoreCols) {
  TaskData data;
  data.rows = 2;
  data.cols = 4;
  data.matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  data.vector = {1.0, 1.0, 1.0, 1.0};
  std::vector<double> expected = {16.0, 20.0};
  ASSERT_TRUE(RunAndCheck(data, expected));
}

TEST(KaurAVertRibbonSchemeTest, NegativeValues) {
  TaskData data;
  data.rows = 2;
  data.cols = 2;
  data.matrix = {-1.0, -2.0, -3.0, -4.0};
  data.vector = {-1.0, -2.0};
  std::vector<double> expected = {7.0, 10.0};
  ASSERT_TRUE(RunAndCheck(data, expected));
}

TEST(KaurAVertRibbonSchemeTest, LargeMatrix) {
  const int size = 100;
  TaskData data;
  data.rows = size;
  data.cols = size;
  data.matrix.resize(static_cast<std::size_t>(size) * size, 1.0);
  data.vector.resize(size, 1.0);
  std::vector<double> expected(size, static_cast<double>(size));
  ASSERT_TRUE(RunAndCheck(data, expected));
}

TEST(KaurAVertRibbonSchemeTest, SingleRow) {
  TaskData data;
  data.rows = 1;
  data.cols = 5;
  data.matrix = {1.0, 2.0, 3.0, 4.0, 5.0};
  data.vector = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> expected = {55.0};
  ASSERT_TRUE(RunAndCheck(data, expected));
}

TEST(KaurAVertRibbonSchemeTest, SingleColumn) {
  TaskData data;
  data.rows = 5;
  data.cols = 1;
  data.matrix = {1.0, 2.0, 3.0, 4.0, 5.0};
  data.vector = {2.0};
  std::vector<double> expected = {2.0, 4.0, 6.0, 8.0, 10.0};
  ASSERT_TRUE(RunAndCheck(data, expected));
}

TEST(KaurAVertRibbonSchemeTest, FloatingPointPrecision) {
  TaskData data;
  data.rows = 2;
  data.cols = 2;
  data.matrix = {0.1, 0.2, 0.3, 0.4};
  data.vector = {0.5, 0.5};
  std::vector<double> expected = {0.2, 0.3};
  ASSERT_TRUE(RunAndCheck(data, expected));
}

static bool ValidationFails(const TaskData &data) {
  KaurAVertRibbonSchemeSEQ task(data);
  return !task.Validation();
}

TEST(KaurAVertRibbonSchemeValidationTest, InvalidRowsZero) {
  TaskData data;
  data.rows = 0;
  data.cols = 2;
  data.matrix = {};
  data.vector = {1.0, 2.0};
  ASSERT_TRUE(ValidationFails(data));
}

TEST(KaurAVertRibbonSchemeValidationTest, InvalidColsZero) {
  TaskData data;
  data.rows = 2;
  data.cols = 0;
  data.matrix = {};
  data.vector = {};
  ASSERT_TRUE(ValidationFails(data));
}

TEST(KaurAVertRibbonSchemeValidationTest, InvalidMatrixSize) {
  TaskData data;
  data.rows = 2;
  data.cols = 2;
  data.matrix = {1.0, 2.0, 3.0};
  data.vector = {1.0, 2.0};
  ASSERT_TRUE(ValidationFails(data));
}

TEST(KaurAVertRibbonSchemeValidationTest, InvalidVectorSize) {
  TaskData data;
  data.rows = 2;
  data.cols = 2;
  data.matrix = {1.0, 2.0, 3.0, 4.0};
  data.vector = {1.0};
  ASSERT_TRUE(ValidationFails(data));
}

}  // namespace
}  // namespace kaur_a_vert_ribbon_scheme
