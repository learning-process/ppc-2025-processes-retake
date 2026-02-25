#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "salykina_a_sparse_matrix_mult/common/include/common.hpp"
#include "salykina_a_sparse_matrix_mult/mpi/include/ops_mpi.hpp"
#include "salykina_a_sparse_matrix_mult/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace salykina_a_sparse_matrix_mult {

static SparseMatrixCRS CreateSparseMatrix(int size, [[maybe_unused]] double sparsity) {
  SparseMatrixCRS crs;
  crs.num_rows = size;
  crs.num_cols = size;
  crs.row_ptr.push_back(0);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (i == j || (i < size - 1 && j == i + 1) || (j < size - 1 && i == j + 1)) {
        crs.values.push_back(static_cast<double>((i * size) + j + 1));
        crs.col_indices.push_back(j);
      }
    }

    crs.row_ptr.push_back(static_cast<int>(crs.values.size()));
  }

  return crs;
}

class SalykinaAMultMatrixRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    int matrix_size = 10000;
    input_data_.matrix_a = CreateSparseMatrix(matrix_size, 0.1);
    input_data_.matrix_b = CreateSparseMatrix(matrix_size, 0.1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.num_rows != input_data_.matrix_a.num_rows) {
      return false;
    }
    if (output_data.num_cols != input_data_.matrix_b.num_cols) {
      return false;
    }
    if (output_data.values.size() != output_data.col_indices.size()) {
      return false;
    }
    if (output_data.row_ptr.size() != static_cast<size_t>(static_cast<size_t>(output_data.num_rows) + 1U)) {
      return false;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

namespace {

TEST_P(SalykinaAMultMatrixRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}
const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SalykinaASparseMatrixMultMPI, SalykinaASparseMatrixMultSEQ>(
        PPC_SETTINGS_salykina_a_sparse_matrix_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = SalykinaAMultMatrixRunPerfTestProcesses::CustomPerfTestName;
INSTANTIATE_TEST_SUITE_P(RunModeTests, SalykinaAMultMatrixRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace salykina_a_sparse_matrix_mult
