#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

#include "savva_d_zeidel_method/common/include/common.hpp"
#include "savva_d_zeidel_method/mpi/include/ops_mpi.hpp"
#include "savva_d_zeidel_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
// #include "util/include/util.hpp"

namespace savva_d_zeidel_method {

class SavvaDZeidelPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data;
  OutType right_output_data;

  void SetUp() override {
    const int n = 6000;

    input_data.n = n;
    input_data.a.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    input_data.b.assign(n, 0.0);
    right_output_data.resize(n);

    for (int i = 0; i < n; ++i) {
      right_output_data[i] = std::sin(0.001 * i) + 2.0;
    }

    for (int i = 0; i < n; ++i) {
      double diag_sum = 0.0;

      if (i > 0) {
        input_data.a[(i * n) + (i - 1)] = -1.0;
        diag_sum += 1.0;
      }

      if (i < n - 1) {
        input_data.a[(i * n) + (i + 1)] = -1.0;
        diag_sum += 1.0;
      }

      input_data.a[(i * n) + i] = diag_sum + 10.0;
    }

    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += input_data.a[(i * n) + j] * right_output_data[j];
      }
      input_data.b[i] = sum;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != right_output_data.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - right_output_data[i]) > 0.0001) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

// Тест на производительность
TEST_P(SavvaDZeidelPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());  // pipeline (SEQ или MPI)
}

// Создаем список всех перф-задач (SEQ и MPI)
const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SavvaDZeidelMPI, SavvaDZeidelSEQ>(PPC_SETTINGS_savva_d_zeidel_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SavvaDZeidelPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SavvaDZeidelPerfTest, kGtestValues, kPerfTestName);

}  // namespace savva_d_zeidel_method
