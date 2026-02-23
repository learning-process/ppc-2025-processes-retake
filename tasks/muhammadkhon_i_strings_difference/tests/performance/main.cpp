#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <utility>

#include "muhammadkhon_i_strings_difference/common/include/common.hpp"
#include "muhammadkhon_i_strings_difference/mpi/include/ops_mpi.hpp"
#include "muhammadkhon_i_strings_difference/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace muhammadkhon_i_strings_difference {

using InType = std::pair<std::string, std::string>;
using OutType = int;

class MuhammadkhonIStringsDifferencePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr size_t kSize = 200000000;

  InType input_data;
  OutType expected_result{};

  void SetUp() override {
    std::string s1(kSize, 'a');
    std::string s2(kSize, 'a');

    for (size_t i = 0; i < kSize; i += 10000000) {
      s2[i] = 'b';
    }
    input_data = std::make_pair(std::move(s1), std::move(s2));

    expected_result = static_cast<OutType>(kSize / 10000000);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_result == output_data;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(MuhammadkhonIStringsDifferencePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MuhammadkhonIStringsDifferenceMPI, MuhammadkhonIStringsDifferenceSEQ>(
        PPC_SETTINGS_muhammadkhon_i_strings_difference);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MuhammadkhonIStringsDifferencePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MuhammadkhonIStringsDifferencePerfTests, kGtestValues, kPerfTestName);

}  // namespace muhammadkhon_i_strings_difference
