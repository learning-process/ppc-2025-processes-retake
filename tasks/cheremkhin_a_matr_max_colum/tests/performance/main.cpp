#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "cheremkhin_a_matr_max_colum/common/include/common.hpp"
#include "cheremkhin_a_matr_max_colum/mpi/include/ops_mpi.hpp"
#include "cheremkhin_a_matr_max_colum/seq/include/ops_seq.hpp"
#include "performance/include/performance.hpp"
#include "util/include/perf_test_util.hpp"

namespace cheremkhin_a_matr_max_colum {

class MatrMaxColumRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kRows = 12000;
  static constexpr int kCols = 12000;

  InType input_data_;
  OutType correct_answer_;

  void SetUp() override {
    input_data_ = InType(kRows, std::vector<int>(kCols));
    for (int i = 0; i < kRows; ++i) {
      for (int j = 0; j < kCols; ++j) {
        input_data_[i][j] = (i * 1000) + j;
      }
    }

    correct_answer_.resize(kCols);
    for (int j = 0; j < kCols; ++j) {
      correct_answer_[j] = ((kRows - 1) * 1000) + j;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return correct_answer_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  void SetPerfAttributes(ppc::performance::PerfAttr &perf_attrs) override {
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attrs.current_timer = [t0] {
      auto now = std::chrono::high_resolution_clock::now();
      auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
      return static_cast<double>(ns) * 1e-9;
    };
    perf_attrs.num_running = 3;
  }
};

TEST_P(MatrMaxColumRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, CheremkhinAMatrMaxColumMPI, CheremkhinAMatrMaxColumSEQ>(
    PPC_SETTINGS_cheremkhin_a_matr_max_colum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MatrMaxColumRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MatrMaxColumRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace cheremkhin_a_matr_max_colum
