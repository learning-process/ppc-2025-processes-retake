#include <gtest/gtest.h>

#include "sokolov_k_min_val_matrix/common/include/common.hpp"
#include "sokolov_k_min_val_matrix/mpi/include/ops_mpi.hpp"
#include "sokolov_k_min_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace k_sokolov_min_val_matrix {

class SokolovKMinValMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 5000;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == 1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SokolovKMinValMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SokolovKMinValMatrixMPI, SokolovKMinValMatrixSEQ>(
    PPC_SETTINGS_sokolov_k_min_val_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SokolovKMinValMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SokolovKMinValMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace k_sokolov_min_val_matrix
