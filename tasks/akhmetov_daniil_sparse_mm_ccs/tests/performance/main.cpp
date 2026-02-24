#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "akhmetov_daniil_sparse_mm_ccs/common/include/common.hpp"
#include "akhmetov_daniil_sparse_mm_ccs/mpi/include/ops_mpi.hpp"
#include "akhmetov_daniil_sparse_mm_ccs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace akhmetov_daniil_sparse_mm_ccs {

using ppc::util::PerfTestParam;

class SparseCCSPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType test_input_data;
  bool data_prepared = false;
  int world_size = 1;
  int rank = 0;
  bool is_seq_test = false;

  static constexpr int kSize = 100000;

  void SetUp() override {
    std::string task_name = std::get<1>(GetParam());
    is_seq_test = (task_name.find("seq") != std::string::npos);

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    PrepareTestData();
  }

  void PrepareTestData() {
    if (data_prepared) {
      return;
    }

    test_input_data.clear();
    test_input_data.push_back(MakeDiagonal(kSize, 1.0));
    test_input_data.push_back(MakeDiagonal(kSize, 2.0));

    data_prepared = true;
  }

  InType GetTestInputData() override {
    return test_input_data;
  }

  bool CheckTestOutputData(OutType &out) override {
    std::string task_name = std::get<1>(GetParam());
    bool is_mpi = (task_name.find("mpi") != std::string::npos);

    if (is_mpi) {
      if (rank != 0) {
        return out.values.empty() && out.row_indices.empty() && out.col_ptr.empty();
      }
    }

    if (rank == 0) {
      EXPECT_EQ(out.rows, kSize);
      EXPECT_EQ(out.cols, kSize);
      EXPECT_EQ(out.col_ptr.size(), static_cast<size_t>(kSize + 1));
      EXPECT_EQ(out.values.size(), out.row_indices.size());

      if (out.col_ptr.empty()) {
        return false;
      }
      if (out.col_ptr[0] != 0) {
        return false;
      }
      for (std::size_t i = 0; i + 1 < out.col_ptr.size(); ++i) {
        if (out.col_ptr[i] > out.col_ptr[i + 1]) {
          return false;
        }
      }
      if (static_cast<std::size_t>(out.col_ptr.back()) != out.values.size()) {
        return false;
      }

      if (!std::ranges::all_of(out.row_indices, [rows = out.rows](int r) { return r >= 0 && r < rows; })) {
        return false;
      }

      return true;
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

namespace {
const auto kPerfTasksTuples =
    ppc::util::MakeAllPerfTasks<InType, SparseMatrixMultiplicationCCSSeq, SparseMatrixMultiplicationCCSMPI>(
        PPC_SETTINGS_akhmetov_daniil_sparse_mm_ccs);

const auto kPerfValues = ppc::util::TupleToGTestValues(kPerfTasksTuples);
const auto kPerfNamePrinter = SparseCCSPerfTest::CustomPerfTestName;

TEST_P(SparseCCSPerfTest, SparseCCSPerformance) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(SparseCCSPerf, SparseCCSPerfTest, kPerfValues, kPerfNamePrinter);

}  // namespace
}  // namespace akhmetov_daniil_sparse_mm_ccs
