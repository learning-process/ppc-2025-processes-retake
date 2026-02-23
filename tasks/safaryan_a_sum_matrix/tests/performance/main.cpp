#include <gtest/gtest.h>

#include <vector>

#include "safaryan_a_sum_matrix/common/include/common.hpp"
#include "safaryan_a_sum_matrix/mpi/include/ops_mpi.hpp"
#include "safaryan_a_sum_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace safaryan_a_sum_matrix {

class SafaryanASumMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 1000;
  InType input_data_;
  OutType expected_result_;

  void SetUp() override {
    input_data_.resize(kMatrixSize_, std::vector<int>(kMatrixSize_, 1));
    expected_result_.resize(kMatrixSize_, kMatrixSize_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_result_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SafaryanASumMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SafaryanASumMatrixMPI, SafaryanASumMatrixSEQ>(
    PPC_SETTINGS_safaryan_a_sum_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SafaryanASumMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SafaryanASumMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace safaryan_a_sum_matrix

// namespace safaryan_a_sum_matrix
