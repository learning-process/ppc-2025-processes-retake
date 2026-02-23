#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zyuzin_n_sum_elements_of_matrix/common/include/common.hpp"
#include "zyuzin_n_sum_elements_of_matrix/mpi/include/ops_mpi.hpp"
#include "zyuzin_n_sum_elements_of_matrix/seq/include/ops_seq.hpp"

namespace zyuzin_n_sum_elements_of_matrix {

class ZyuzinNSumElementsOfMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_sum_{0.0};

  void SetUp() override {
    int rows = 10000;
    int cols = 10000;
    std::vector<double> data(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
      data[i] = static_cast<double>(i + 1);
    }

    double expected_sum = static_cast<double>(rows) * cols * (static_cast<double>(rows) * cols + 1) / 2.0;
    input_data_ = std::make_tuple(rows, cols, std::move(data));
    expected_sum_ = expected_sum;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (expected_sum_ == 0.0) {
      return output_data == 0.0;
    }
    const double k_eps = 1e-9;
    double relative_error = std::abs(output_data - expected_sum_) / std::abs(expected_sum_);
    return relative_error < k_eps;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZyuzinNSumElementsOfMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ZyuzinNSumElementsOfMatrixMPI, ZyuzinNSumElementsOfMatrixSEQ>(
        PPC_SETTINGS_zyuzin_n_sum_elements_of_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZyuzinNSumElementsOfMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZyuzinNSumElementsOfMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace zyuzin_n_sum_elements_of_matrix
