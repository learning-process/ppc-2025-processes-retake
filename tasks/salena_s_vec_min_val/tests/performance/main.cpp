#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
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
    return true;  // В тестах производительности проверка результата не так критична
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  void SetPerfAttributes(ppc::performance::PerfAttr &perf_attr) override {
    perf_attr.num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr.current_timer = [t0] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };
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
