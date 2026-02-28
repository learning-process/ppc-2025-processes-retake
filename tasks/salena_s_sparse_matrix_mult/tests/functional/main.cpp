#include <gtest/gtest.h>
#include <vector>
#include <tuple>
#include <string>
#include <random>

#include "salena_s_sparse_matrix_mult/common/include/common.hpp"
#include "salena_s_sparse_matrix_mult/mpi/include/ops_mpi.hpp"
#include "salena_s_sparse_matrix_mult/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace salena_s_sparse_matrix_mult {

using TestType = std::tuple<int, int, int, double, std::string>;

//  static
static SparseMatrixCRS GenSparse(int rows, int cols, double density) {
  SparseMatrixCRS mat;
  mat.rows = rows;
  mat.cols = cols;
  mat.row_ptr.push_back(0);

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> val_dist(-10.0, 10.0);
  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

  int nnz = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (prob_dist(gen) < density) {
        mat.values.push_back(val_dist(gen));
        mat.col_indices.push_back(j);
        nnz++;
      }
    }
    mat.row_ptr.push_back(nnz);
  }
  return mat;
}

class SparseMultFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<4>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int m = std::get<0>(params);
    int k = std::get<1>(params);
    int n = std::get<2>(params);
    double density = std::get<3>(params);

    input_data_.A = GenSparse(m, k, density);
    input_data_.B = GenSparse(k, n, density);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    SparseMatrixCRS expected;
    expected.rows = input_data_.A.rows;
    expected.cols = input_data_.B.cols;
    expected.row_ptr.assign(expected.rows + 1, 0);

    const auto& A = input_data_.A;
    const auto& B = input_data_.B;

    std::vector<int> marker(B.cols, -1);
    std::vector<double> temp_values(B.cols, 0.0);

    for (int i = 0; i < A.rows; ++i) {
      int row_nz = 0;
      std::vector<int> current_row_cols;

      for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
        int a_col = A.col_indices[j];
        double a_val = A.values[j];

        for (int k = B.row_ptr[a_col]; k < B.row_ptr[a_col + 1]; ++k) {
          int b_col = B.col_indices[k];
          double b_val = B.values[k];

          if (marker[b_col] != i) {
            marker[b_col] = i;
            current_row_cols.push_back(b_col);
            temp_values[b_col] = a_val * b_val;
          } else {
            temp_values[b_col] += a_val * b_val;
          }
        }
      }

      std::sort(current_row_cols.begin(), current_row_cols.end());
      for (int col : current_row_cols) {
        if (temp_values[col] != 0.0) { 
          expected.values.push_back(temp_values[col]);
          expected.col_indices.push_back(col);
          row_nz++;
        }
      }
      expected.row_ptr[i + 1] = expected.row_ptr[i] + row_nz;
    }

    if (expected.row_ptr != output_data.row_ptr) return false;
    if (expected.col_indices != output_data.col_indices) return false;
    
    for(size_t i = 0; i < expected.values.size(); ++i) {
        if (std::abs(expected.values[i] - output_data.values[i]) > 1e-4) return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(SparseMultFuncTests, RunMult) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(10, 10, 10, 0.2, "10x10_low"),
    std::make_tuple(20, 30, 20, 0.1, "20x30_med"),
    std::make_tuple(50, 50, 50, 0.05, "50x50_low")
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SparseMatrixMultMPI, InType>(kTestParam, PPC_SETTINGS_salena_s_sparse_matrix_mult),
                   ppc::util::AddFuncTask<SparseMatrixMultSeq, InType>(kTestParam, PPC_SETTINGS_salena_s_sparse_matrix_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = SparseMultFuncTests::PrintFuncTestName<SparseMultFuncTests>;
INSTANTIATE_TEST_SUITE_P(SparseMultTests, SparseMultFuncTests, kGtestValues, kPerfTestName);

}  // namespace salena_s_sparse_matrix_mult