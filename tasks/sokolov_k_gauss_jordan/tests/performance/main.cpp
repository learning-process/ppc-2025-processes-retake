#include <gtest/gtest.h>

#include "sokolov_k_gauss_jordan/common/include/common.hpp"
#include "sokolov_k_gauss_jordan/mpi/include/ops_mpi.hpp"
#include "sokolov_k_gauss_jordan/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sokolov_k_gauss_jordan {

class SokolovKGaussJordanPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_ * 9;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SokolovKGaussJordanPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SokolovKGaussJordanMPI, SokolovKGaussJordanSEQ>(
    PPC_SETTINGS_sokolov_k_gauss_jordan);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SokolovKGaussJordanPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SokolovKGaussJordanPerfTests, kGtestValues, kPerfTestName);

}  // namespace sokolov_k_gauss_jordan
