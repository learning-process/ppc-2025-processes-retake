#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "fedoseev_count_words_in_string/common/include/common.hpp"
#include "fedoseev_count_words_in_string/mpi/include/ops_mpi.hpp"
#include "fedoseev_count_words_in_string/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace fedoseev_count_words_in_string {

using InType = fedoseev_count_words_in_string::InType;
using OutType = fedoseev_count_words_in_string::OutType;

class FedoseevRunPerfTestWordsCount : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr int kRepeat = 300000;

  void SetUp() override {
    std::string unit = "word ";
    input_data_.reserve(unit.size() * static_cast<std::size_t>(kRepeat));
    for (int i = 0; i < kRepeat; ++i) {
      input_data_ += unit;
    }
    expected_ = kRepeat;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_{};
};

TEST_P(FedoseevRunPerfTestWordsCount, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, fedoseev_count_words_in_string::FedoseevCountWordsInStringMPI,
                                fedoseev_count_words_in_string::FedoseevCountWordsInStringSEQ>(
        PPC_SETTINGS_fedoseev_count_words_in_string);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = FedoseevRunPerfTestWordsCount::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, FedoseevRunPerfTestWordsCount, kGtestValues, kPerfTestName);

}  // namespace fedoseev_count_words_in_string
