#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <random>
#include <vector>

#include "salena_s_sparse_matrix_mult/common/include/common.hpp"
#include "salena_s_sparse_matrix_mult/mpi/include/ops_mpi.hpp"
#include "salena_s_sparse_matrix_mult/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace salena_s_sparse_matrix_mult {

namespace {
SparseMatrixCRS GenSparsePerf(int rows, int cols, double density) {
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
}  // namespace

class SparseMultPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_.A = GenSparsePerf(200, 200, 0.05);
    input_data_.B = GenSparsePerf(200, 200, 0.05);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  void SetPerfAttributes(ppc::performance::PerfAttr &perf_attr) override {
    perf_attr.num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr.current_timer = [t0] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };
  }

 private:
  InType input_data_;
};

TEST_P(SparseMultPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SparseMatrixMultMPI, SparseMatrixMultSeq>(
    PPC_SETTINGS_salena_s_sparse_matrix_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = SparseMultPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(SparsePerfTests, SparseMultPerfTests, kGtestValues, kPerfTestName);

}  // namespace salena_s_sparse_matrix_mult
