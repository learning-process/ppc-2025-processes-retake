#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "salykina_a_sparse_matrix_mult/common/include/common.hpp"
#include "salykina_a_sparse_matrix_mult/mpi/include/ops_mpi.hpp"
#include "salykina_a_sparse_matrix_mult/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace salykina_a_sparse_matrix_mult {

static SparseMatrixCRS DenseToCRS(const std::vector<std::vector<double>> &dense) {
  SparseMatrixCRS crs;
  crs.num_rows = static_cast<int>(dense.size());
  crs.num_cols = dense.empty() ? 0 : static_cast<int>(dense[0].size());
  crs.row_ptr.push_back(0);

  for (size_t i = 0; i < dense.size(); i++) {
    for (size_t j = 0; j < dense[i].size(); j++) {
      if (std::abs(dense[i][j]) > 1e-10) {
        crs.values.push_back(dense[i][j]);
        crs.col_indices.push_back(static_cast<int>(j));
      }
    }

    crs.row_ptr.push_back(static_cast<int>(crs.values.size()));
  }

  return crs;
}

static std::vector<std::vector<double>> CRSToDense(const SparseMatrixCRS &crs) {
  std::vector<std::vector<double>> dense(crs.num_rows, std::vector<double>(crs.num_cols, 0.0));

  for (int i = 0; i < crs.num_rows; i++) {
    int row_start = crs.row_ptr[i];
    int row_end = crs.row_ptr[i + 1];
    for (int j = row_start; j < row_end; j++) {
      dense[i][crs.col_indices[j]] = crs.values[j];
    }
  }

  return dense;
}

static std::vector<std::vector<double>> MultiplyDense(const std::vector<std::vector<double>> &a,
                                                      const std::vector<std::vector<double>> &b) {
  int rows_a = static_cast<int>(a.size());
  int cols_a = a.empty() ? 0 : static_cast<int>(a[0].size());
  int cols_b = b.empty() ? 0 : static_cast<int>(b[0].size());

  std::vector<std::vector<double>> result(rows_a, std::vector<double>(cols_b, 0.0));
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      for (int k = 0; k < cols_a; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return result;
}

class SalykinaASparseMatrixMultRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int matrix_size = std::get<0>(params);
    std::vector<std::vector<double>> dense_a(matrix_size, std::vector<double>(matrix_size, 0.0));
    std::vector<std::vector<double>> dense_b(matrix_size, std::vector<double>(matrix_size, 0.0));

    for (int i = 0; i < matrix_size; i++) {
      dense_a[i][i] = static_cast<double>(i + 1);

      if (i + 1 < matrix_size) {
        dense_a[i][i + 1] = 0.5;
      }
      if (i > 0) {
        dense_a[i][i - 1] = 0.3;
      }
    }

    for (int i = 0; i < matrix_size; i++) {
      dense_b[i][i] = static_cast<double>(matrix_size - i);

      if (i + 1 < matrix_size) {
        dense_b[i][i + 1] = 0.2;
      }
    }

    input_data_.matrix_a = DenseToCRS(dense_a);
    input_data_.matrix_b = DenseToCRS(dense_b);

    std::vector<std::vector<double>> expected_dense = MultiplyDense(dense_a, dense_b);
    expected_result_ = DenseToCRS(expected_dense);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<std::vector<double>> output_dense = CRSToDense(output_data);
    std::vector<std::vector<double>> expected_dense = CRSToDense(expected_result_);

    if (output_dense.size() != expected_dense.size()) {
      return false;
    }
    if (output_dense.empty()) {
      return expected_dense.empty();
    }
    if (output_dense[0].size() != expected_dense[0].size()) {
      return false;
    }

    const double epsilon = 1e-6;
    for (size_t i = 0; i < output_dense.size(); i++) {
      for (size_t j = 0; j < output_dense[i].size(); j++) {
        if (std::abs(output_dense[i][j] - expected_dense[i][j]) > epsilon) {
          return false;
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  SparseMatrixCRS expected_result_;
};

namespace {

TEST_P(SalykinaASparseMatrixMultRunFuncTestsProcesses, SparseMatrixMultiplication) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(4, "4"), std::make_tuple(8, "8"),
                                            std::make_tuple(16, "16")};
const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SalykinaASparseMatrixMultMPI, InType>(
                                               kTestParam, PPC_SETTINGS_salykina_a_sparse_matrix_mult),
                                           ppc::util::AddFuncTask<SalykinaASparseMatrixMultSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_salykina_a_sparse_matrix_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName =
    SalykinaASparseMatrixMultRunFuncTestsProcesses::PrintFuncTestName<SalykinaASparseMatrixMultRunFuncTestsProcesses>;
INSTANTIATE_TEST_SUITE_P(SparseMatrixTests, SalykinaASparseMatrixMultRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace salykina_a_sparse_matrix_mult
