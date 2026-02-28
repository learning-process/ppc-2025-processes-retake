#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "yusupkina_m_seidel_method/common/include/common.hpp"
#include "yusupkina_m_seidel_method/mpi/include/ops_mpi.hpp"
#include "yusupkina_m_seidel_method/seq/include/ops_seq.hpp"

namespace yusupkina_m_seidel_method {

class YusupkinaMSeidelMethodPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data;
  OutType expected_solution;

  void SetUp() override {
    const int n = 8000;

    std::vector<double> matrix(static_cast<size_t>(n) * n, 0.0);
    std::vector<double> rhs(n, 0.0);
    expected_solution.resize(n);
    for (int i = 0; i < n; ++i) {
      expected_solution[i] = std::sin(0.001 * i) + 2.0;
    }
    for (int i = 0; i < n; ++i) {
      double diag_sum = 0.0;
      if (i > 0) {
        matrix[(static_cast<size_t>(i) * n) + (i - 1)] = -1.0;
        diag_sum += 1.0;
      }
      if (i < n - 1) {
        matrix[(static_cast<size_t>(i) * n) + (i + 1)] = -1.0;
        diag_sum += 1.0;
      }
      matrix[(static_cast<size_t>(i) * n) + i] = diag_sum + 10.0;
    }

    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += matrix[(static_cast<size_t>(i) * n) + j] * expected_solution[j];
      }
      rhs[i] = sum;
    }

    input_data = InType{.matrix = matrix, .rhs = rhs, .n = n};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double epsilon = 1e-4;

    if (output_data.size() != expected_solution.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); i++) {
      if (std::abs(output_data[i] - expected_solution[i]) > epsilon) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(YusupkinaMSeidelMethodPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, YusupkinaMSeidelMethodMPI, YusupkinaMSeidelMethodSEQ>(
    PPC_SETTINGS_yusupkina_m_seidel_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = YusupkinaMSeidelMethodPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, YusupkinaMSeidelMethodPerfTests, kGtestValues, kPerfTestName);

}  // namespace yusupkina_m_seidel_method
