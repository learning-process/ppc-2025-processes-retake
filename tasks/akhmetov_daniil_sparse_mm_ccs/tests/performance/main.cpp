#include <gtest/gtest.h>

#include <cstdlib>
#include <vector>

#include "akhmetov_daniil_sparse_mm_ccs/common/include/common.hpp"
#include "akhmetov_daniil_sparse_mm_ccs/mpi/include/ops_mpi.hpp"
#include "akhmetov_daniil_sparse_mm_ccs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace akhmetov_daniil_sparse_mm_ccs {

class SparseCCSPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType test_input_data;
  const int k_size = 2000;

  void SetUp() override {
    test_input_data.clear();
    test_input_data.push_back(MakeDiagonal(k_size, 1.0));
    test_input_data.push_back(MakeDiagonal(k_size, 2.0));
  }

  InType GetTestInputData() override {
    return test_input_data;
  }

  bool CheckTestOutputData(OutType &out) override {
    if (out.rows == 0 || out.cols == 0) {
      return false;
    }

    if (out.rows != k_size || out.cols != k_size) {
      return false;
    }

    if (out.values.empty()) {
      return false;
    }

    if (std::abs(out.values[0] - 2.0) > 1e-9) {
      return false;
    }

    return true;
  }

 private:
  static SparseMatrixCCS MakeDiagonal(int n, double v) {
    SparseMatrixCCS m;
    m.rows = n;
    m.cols = n;
    m.col_ptr.resize(n + 1);
    for (int j = 0; j < n; ++j) {
      m.col_ptr[j] = static_cast<int>(m.values.size());
      m.values.push_back(v);
      m.row_indices.push_back(j);
    }
    m.col_ptr[n] = static_cast<int>(m.values.size());
    return m;
  }
};

TEST_P(SparseCCSPerfTest, SparseCCSPerformance) {
  ExecuteTest(GetParam());
}

namespace {

const auto kPerfTasksTuples =
    ppc::util::MakeAllPerfTasks<InType, SparseMatrixMultiplicationCCSSeq, SparseMatrixMultiplicationCCSMPI>(
        PPC_SETTINGS_akhmetov_daniil_sparse_mm_ccs);

const auto kPerfValues = ppc::util::TupleToGTestValues(kPerfTasksTuples);
const auto kPerfNamePrinter = SparseCCSPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(SparseCCSPerf, SparseCCSPerfTest, kPerfValues, kPerfNamePrinter);

}  // namespace
}  // namespace akhmetov_daniil_sparse_mm_ccs
