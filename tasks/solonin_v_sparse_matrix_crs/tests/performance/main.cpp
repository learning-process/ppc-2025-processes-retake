#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "solonin_v_sparse_matrix_crs/common/include/common.hpp"
#include "solonin_v_sparse_matrix_crs/mpi/include/ops_mpi.hpp"
#include "solonin_v_sparse_matrix_crs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace solonin_v_sparse_matrix_crs {

class SoloninVCRSPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr int kSize = 50000;

 protected:
  void SetUp() override {
    BuildDiagonal(vals_a_, cols_a_, ptr_a_, kSize, 1.0);
    BuildDiagonal(vals_b_, cols_b_, ptr_b_, kSize, 2.0);
  }

  bool CheckTestOutputData(OutType &out) final {
    auto &[vals, cols, ptr] = out;
    if (ptr.empty()) return true;
    if (ptr[0] != 0) return false;
    if (vals.size() != cols.size()) return false;
    for (size_t i = 0; i + 1 < ptr.size(); i++) {
      if (ptr[i] > ptr[i + 1]) return false;
    }
    return true;
  }

  InType GetTestInputData() final {
    return std::make_tuple(vals_a_, cols_a_, ptr_a_, vals_b_, cols_b_, ptr_b_, kSize, kSize, kSize);
  }

 private:
  static void BuildDiagonal(std::vector<double> &vals, std::vector<int> &cols,
                             std::vector<int> &ptr, int n, double val) {
    vals.clear();
    cols.clear();
    ptr.clear();
    ptr.push_back(0);
    for (int i = 0; i < n; i++) {
      vals.push_back(val);
      cols.push_back(i);
      if (i % 10 == 0 && i + 1 < n) {
        vals.push_back(0.5);
        cols.push_back(i + 1);
      }
      ptr.push_back(static_cast<int>(vals.size()));
    }
  }

  std::vector<double> vals_a_, vals_b_;
  std::vector<int> cols_a_, cols_b_;
  std::vector<int> ptr_a_, ptr_b_;
};

TEST_P(SoloninVCRSPerfTests, RunPerfModes) { ExecuteTest(GetParam()); }

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SoloninVSparseMulCRSMPI, SoloninVSparseMulCRSSEQ>(
    PPC_SETTINGS_solonin_v_sparse_matrix_crs);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = SoloninVCRSPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SoloninVCRSPerfTests, kGtestValues, kPerfTestName);

}  // namespace solonin_v_sparse_matrix_crs
