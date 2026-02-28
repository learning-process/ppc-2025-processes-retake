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
  const int k_size = 2000;
  InType test_input_data;

  void SetUp() override {
    SparseMatrixCCS a = MakeDiagonal(k_size, 1.0);
    SparseMatrixCCS b = MakeDiagonal(k_size, 2.0);

    test_input_data.clear();
    test_input_data.push_back(std::move(a));
    test_input_data.push_back(std::move(b));
  }

  InType GetTestInputData() override {
    return test_input_data;
  }

  bool CheckTestOutputData(OutType &out) override {
    if (out.rows != k_size || out.cols != k_size) {
      return false;
    }

    if (out.values.empty() || out.row_indices.empty() || out.col_ptr.empty()) {
      return false;
    }

    for (int i = 0; i < k_size; ++i) {
      bool found_diagonal = false;
      for (size_t j = out.col_ptr[i]; j < out.col_ptr[i + 1]; ++j) {
        if (out.row_indices[j] == i) {
          if (std::abs(out.values[j] - 2.0) > 1e-9) {
            return false;
          }
          found_diagonal = true;
        } else {
          return false;
        }
      }
      if (!found_diagonal) {
        return false;
      }
    }

    return true;
  }

 private:
  static SparseMatrixCCS MakeDiagonal(int n, double v) {
    SparseMatrixCCS m;
    m.rows = n;
    m.cols = n;
    m.col_ptr.resize(n + 1, 0);

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
    ppc::util::MakeAllPerfTasks<InType, SparseMatrixMultiplicationCCSMPI, SparseMatrixMultiplicationCCSSeq>(
        PPC_SETTINGS_akhmetov_daniil_sparse_mm_ccs);

const auto kPerfValues = ppc::util::TupleToGTestValues(kPerfTasksTuples);
const auto kPerfNamePrinter = SparseCCSPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(SparseCCSPerf, SparseCCSPerfTest, kPerfValues, kPerfNamePrinter);
}  // namespace

}  // namespace akhmetov_daniil_sparse_mm_ccs
