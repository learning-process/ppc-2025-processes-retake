#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

#include "rysev_m_matrix_multiple/common/include/common.hpp"
#include "rysev_m_matrix_multiple/mpi/include/ops_mpi.hpp"
#include "rysev_m_matrix_multiple/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace rysev_m_matrix_multiple {

class RysevMRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kSize_ = 100;
  InType input_data_;

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 5);

    std::vector<int> a(static_cast<size_t>(kSize_) * kSize_);
    std::vector<int> b(static_cast<size_t>(kSize_) * kSize_);

    for (int i = 0; i < kSize_ * kSize_; ++i) {
      a[i] = dis(gen);
      b[i] = dis(gen);
    }

    input_data_ = std::make_tuple(a, b, kSize_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RysevMRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RysevMMatrMulMPI, RysevMMatrMulSEQ>(PPC_SETTINGS_rysev_m_matrix_multiple);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RysevMRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RysevMRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace rysev_m_matrix_multiple
