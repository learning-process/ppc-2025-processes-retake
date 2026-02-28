#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "vinyaikina_e_matrix_sum/common/include/common.hpp"
#include "vinyaikina_e_matrix_sum/mpi/include/ops_mpi.hpp"
#include "vinyaikina_e_matrix_sum/seq/include/ops_seq.hpp"

namespace vinyaikina_e_matrix_sum {

class VinyaikinaEMatrixSumFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);

    int rows = size;
    int cols = size;
    input_data_.resize((static_cast<size_t>(rows) * static_cast<size_t>(cols)) + 2);
    input_data_[0] = rows;
    input_data_[1] = cols;

    expected_output_.resize(rows);
    for (int i = 0; i < rows; i++) {
      int row_sum = 0;
      for (int j = 0; j < cols; j++) {
        int val = (i * cols) + j + 1;
        input_data_[2 + (i * cols) + j] = val;
        row_sum += val;
      }
      expected_output_[i] = row_sum;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(VinyaikinaEMatrixSumFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<VinyaikinaEMatrixSumMPI, InType>(kTestParam, PPC_SETTINGS_vinyaikina_e_matrix_sum),
    ppc::util::AddFuncTask<VinyaikinaEMatrixSumSEQ, InType>(kTestParam, PPC_SETTINGS_vinyaikina_e_matrix_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VinyaikinaEMatrixSumFuncTests::PrintFuncTestName<VinyaikinaEMatrixSumFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, VinyaikinaEMatrixSumFuncTests, kGtestValues, kPerfTestName);

template <typename TaskType>
void ValidateMatrixSum(const InType &input, const OutType &expected) {
  auto task = std::make_shared<TaskType>(input);
  EXPECT_TRUE(task->Validation());
  EXPECT_TRUE(task->PreProcessing());
  EXPECT_TRUE(task->Run());
  EXPECT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), expected);
}

TEST(VinyaikinaEMatrixSumEdge, SingleElementSEQ) {
  ValidateMatrixSum<VinyaikinaEMatrixSumSEQ>({1, 1, 42}, {42});
}

TEST(VinyaikinaEMatrixSumEdge, SingleElementMPI) {
  ValidateMatrixSum<VinyaikinaEMatrixSumMPI>({1, 1, 42}, {42});
}

TEST(VinyaikinaEMatrixSumEdge, SingleRowSEQ) {
  ValidateMatrixSum<VinyaikinaEMatrixSumSEQ>({1, 5, 1, 2, 3, 4, 5}, {15});
}

TEST(VinyaikinaEMatrixSumEdge, SingleRowMPI) {
  ValidateMatrixSum<VinyaikinaEMatrixSumMPI>({1, 5, 1, 2, 3, 4, 5}, {15});
}

TEST(VinyaikinaEMatrixSumEdge, SingleColumnSEQ) {
  ValidateMatrixSum<VinyaikinaEMatrixSumSEQ>({5, 1, 10, 20, 30, 40, 50}, {10, 20, 30, 40, 50});
}

TEST(VinyaikinaEMatrixSumEdge, SingleColumnMPI) {
  ValidateMatrixSum<VinyaikinaEMatrixSumMPI>({5, 1, 10, 20, 30, 40, 50}, {10, 20, 30, 40, 50});
}

TEST(VinyaikinaEMatrixSumEdge, AllZerosSEQ) {
  ValidateMatrixSum<VinyaikinaEMatrixSumSEQ>({3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0});
}

TEST(VinyaikinaEMatrixSumEdge, AllZerosMPI) {
  ValidateMatrixSum<VinyaikinaEMatrixSumMPI>({3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0});
}

TEST(VinyaikinaEMatrixSumEdge, NegativeValuesSEQ) {
  ValidateMatrixSum<VinyaikinaEMatrixSumSEQ>({2, 3, -1, -2, -3, -4, -5, -6}, {-6, -15});
}

TEST(VinyaikinaEMatrixSumEdge, NegativeValuesMPI) {
  ValidateMatrixSum<VinyaikinaEMatrixSumMPI>({2, 3, -1, -2, -3, -4, -5, -6}, {-6, -15});
}

TEST(VinyaikinaEMatrixSumEdge, MixedValuesSEQ) {
  ValidateMatrixSum<VinyaikinaEMatrixSumSEQ>({2, 3, 1, -2, 3, -4, 5, -6}, {2, -5});
}

TEST(VinyaikinaEMatrixSumEdge, MixedValuesMPI) {
  ValidateMatrixSum<VinyaikinaEMatrixSumMPI>({2, 3, 1, -2, 3, -4, 5, -6}, {2, -5});
}

TEST(VinyaikinaEMatrixSumEdge, NonSquareWiderSEQ) {
  ValidateMatrixSum<VinyaikinaEMatrixSumSEQ>({2, 4, 1, 2, 3, 4, 5, 6, 7, 8}, {10, 26});
}

TEST(VinyaikinaEMatrixSumEdge, NonSquareWiderMPI) {
  ValidateMatrixSum<VinyaikinaEMatrixSumMPI>({2, 4, 1, 2, 3, 4, 5, 6, 7, 8}, {10, 26});
}

TEST(VinyaikinaEMatrixSumEdge, NonSquareTallerSEQ) {
  ValidateMatrixSum<VinyaikinaEMatrixSumSEQ>({4, 2, 1, 2, 3, 4, 5, 6, 7, 8}, {3, 7, 11, 15});
}

TEST(VinyaikinaEMatrixSumEdge, NonSquareTallerMPI) {
  ValidateMatrixSum<VinyaikinaEMatrixSumMPI>({4, 2, 1, 2, 3, 4, 5, 6, 7, 8}, {3, 7, 11, 15});
}

TEST(VinyaikinaEMatrixSumEdge, LargeMatrixSEQ) {
  int rows = 50;
  int cols = 60;
  InType input((static_cast<size_t>(rows) * cols) + 2);
  input[0] = rows;
  input[1] = cols;
  OutType expected(rows);
  for (int i = 0; i < rows; i++) {
    int row_sum = 0;
    for (int j = 0; j < cols; j++) {
      int val = (((i * cols) + j) % 100) + 1;
      input[2 + (i * cols) + j] = val;
      row_sum += val;
    }
    expected[i] = row_sum;
  }
  ValidateMatrixSum<VinyaikinaEMatrixSumSEQ>(input, expected);
}

TEST(VinyaikinaEMatrixSumEdge, LargeMatrixMPI) {
  int rows = 50;
  int cols = 60;
  InType input((static_cast<size_t>(rows) * cols) + 2);
  input[0] = rows;
  input[1] = cols;
  OutType expected(rows);
  for (int i = 0; i < rows; i++) {
    int row_sum = 0;
    for (int j = 0; j < cols; j++) {
      int val = (((i * cols) + j) % 100) + 1;
      input[2 + (i * cols) + j] = val;
      row_sum += val;
    }
    expected[i] = row_sum;
  }
  ValidateMatrixSum<VinyaikinaEMatrixSumMPI>(input, expected);
}

TEST(VinyaikinaEMatrixSumEdge, AllOnesSEQ) {
  int rows = 4;
  int cols = 4;
  InType input((static_cast<size_t>(rows) * cols) + 2, 1);
  input[0] = rows;
  input[1] = cols;
  OutType expected(rows, cols);
  ValidateMatrixSum<VinyaikinaEMatrixSumSEQ>(input, expected);
}

TEST(VinyaikinaEMatrixSumEdge, AllOnesMPI) {
  int rows = 4;
  int cols = 4;
  InType input((static_cast<size_t>(rows) * cols) + 2, 1);
  input[0] = rows;
  input[1] = cols;
  OutType expected(rows, cols);
  ValidateMatrixSum<VinyaikinaEMatrixSumMPI>(input, expected);
}

TEST(VinyaikinaEMatrixSumEdge, ManyRowsFewColsSEQ) {
  int rows = 10;
  int cols = 2;
  InType input((static_cast<size_t>(rows) * cols) + 2);
  input[0] = rows;
  input[1] = cols;
  OutType expected(rows);
  for (int i = 0; i < rows; i++) {
    input[2 + (i * cols)] = i + 1;
    input[2 + (i * cols) + 1] = (i + 1) * 10;
    expected[i] = (i + 1) + ((i + 1) * 10);
  }
  ValidateMatrixSum<VinyaikinaEMatrixSumSEQ>(input, expected);
}

TEST(VinyaikinaEMatrixSumEdge, ManyRowsFewColsMPI) {
  int rows = 10;
  int cols = 2;
  InType input((static_cast<size_t>(rows) * cols) + 2);
  input[0] = rows;
  input[1] = cols;
  OutType expected(rows);
  for (int i = 0; i < rows; i++) {
    input[2 + (i * cols)] = i + 1;
    input[2 + (i * cols) + 1] = (i + 1) * 10;
    expected[i] = (i + 1) + ((i + 1) * 10);
  }
  ValidateMatrixSum<VinyaikinaEMatrixSumMPI>(input, expected);
}

}  // namespace

}  // namespace vinyaikina_e_matrix_sum
