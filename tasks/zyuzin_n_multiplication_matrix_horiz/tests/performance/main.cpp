#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zyuzin_n_multiplication_matrix_horiz/common/include/common.hpp"
#include "zyuzin_n_multiplication_matrix_horiz/mpi/include/ops_mpi.hpp"
#include "zyuzin_n_multiplication_matrix_horiz/seq/include/ops_seq.hpp"

namespace zyuzin_n_multiplication_matrix_horiz {

class ZyuzinNMultiplicationMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr size_t kCount = 500;
  InType input_data;
  OutType expected;

 protected:
  void SetUp() override {
    matrix_a_ = std::vector<std::vector<double>>(kCount, std::vector<double>(kCount, 0));
    matrix_b_ = std::vector<std::vector<double>>(kCount, std::vector<double>(kCount, 0.1));
    for (size_t i = 0; i < kCount; i++) {
      matrix_a_[i][i] = 1.0;
    }
    expected = std::vector<std::vector<double>>(kCount, std::vector<double>(kCount, 0));
    for (size_t i = 0; i < kCount; ++i) {
      for (size_t j = 0; j < kCount; ++j) {
        expected[i][j] = matrix_a_[i][i] * matrix_b_[i][j];
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected.size() || output_data[0].size() != expected[0].size()) {
      return false;
    }
    const double k_eps = 1e-10;
    for (size_t i = 0; i < expected.size(); i++) {
      for (size_t j = 0; j < expected[0].size(); j++) {
        if (std::fabs(output_data[i][j] - expected[i][j]) > k_eps) {
          return false;
        }
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return std::make_pair(matrix_a_, matrix_b_);
  }

 private:
  std::vector<std::vector<double>> matrix_a_;
  std::vector<std::vector<double>> matrix_b_;
};

TEST_P(ZyuzinNMultiplicationMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ZyuzinNMultiplicationMatrixMPI, ZyuzinNMultiplicationMatrixSEQ>(
        PPC_SETTINGS_zyuzin_n_multiplication_matrix_horiz);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZyuzinNMultiplicationMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZyuzinNMultiplicationMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zyuzin_n_multiplication_matrix_horiz
