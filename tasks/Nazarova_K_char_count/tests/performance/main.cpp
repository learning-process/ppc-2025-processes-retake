#include <gtest/gtest.h>

#include <cstddef>

#include "Nazarova_K_char_count/common/include/common.hpp"
#include "Nazarova_K_char_count/mpi/include/ops_mpi.hpp"
#include "Nazarova_K_char_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nazarova_k_char_count_processes {

class NazarovaKCharCountRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr std::size_t kCount = 1'000'000;
  InType input_data_{};
  OutType expected_{};

  void SetUp() override {
    input_data_.target = 'a';
    input_data_.text.assign(kCount, 'q');
    expected_ = 0;
    for (std::size_t i = 0; i < input_data_.text.size(); i++) {
      if (i % 7 == 0) {
        input_data_.text[i] = input_data_.target;
        expected_++;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(NazarovaKCharCountRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, NazarovaKCharCountMPI, NazarovaKCharCountSEQ>(
    PPC_SETTINGS_Nazarova_K_char_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NazarovaKCharCountRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NazarovaKCharCountRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace nazarova_k_char_count_processes
