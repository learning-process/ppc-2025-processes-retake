#include <gtest/gtest.h>

#include <cstddef>

#include "util/include/perf_test_util.hpp"
#include "vinyaikina_e_matrix_sum/common/include/common.hpp"
#include "vinyaikina_e_matrix_sum/mpi/include/ops_mpi.hpp"
#include "vinyaikina_e_matrix_sum/seq/include/ops_seq.hpp"

namespace vinyaikina_e_matrix_sum {

class VinyaikinaEMatrixSumPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kRows = 5000;
  static constexpr int kCols = 5000;
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    input_data_.resize((static_cast<size_t>(kRows) * kCols) + 2);
    input_data_[0] = kRows;
    input_data_[1] = kCols;
    for (size_t i = 2; i < input_data_.size(); i++) {
      input_data_[i] = 1;
    }
    expected_output_.assign(kRows, kCols);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(VinyaikinaEMatrixSumPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, VinyaikinaEMatrixSumMPI, VinyaikinaEMatrixSumSEQ>(
    PPC_SETTINGS_vinyaikina_e_matrix_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VinyaikinaEMatrixSumPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, VinyaikinaEMatrixSumPerfTests, kGtestValues, kPerfTestName);

}  // namespace vinyaikina_e_matrix_sum
