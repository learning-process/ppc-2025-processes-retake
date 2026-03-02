#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include "salena_s_vec_min_val/common/include/common.hpp"
#include "salena_s_vec_min_val/mpi/include/ops_mpi.hpp"
#include "salena_s_vec_min_val/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace salena_s_vec_min_val {

class VectorMinPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_.resize(static_cast<std::size_t>(kCount_));
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    for (int i = 0; i < kCount_; ++i) {
      input_data_[static_cast<std::size_t>(i)] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int expected_min = *std::min_element(input_data_.begin(), input_data_.end());
    return expected_min == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  const int kCount_ = 1000000;
  InType input_data_{};
};

TEST_P(VectorMinPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TestTaskMPI, TestTaskSEQ>(PPC_SETTINGS_salena_s_vec_min_val);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = VectorMinPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, VectorMinPerfTests, kGtestValues, kPerfTestName);

}  // namespace salena_s_vec_min_val
