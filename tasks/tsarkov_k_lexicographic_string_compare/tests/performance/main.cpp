#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <utility>

#include "tsarkov_k_lexicographic_string_compare/common/include/common.hpp"
#include "tsarkov_k_lexicographic_string_compare/mpi/include/ops_mpi.hpp"
#include "tsarkov_k_lexicographic_string_compare/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tsarkov_k_lexicographic_string_compare {

class TsarkovKRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const std::size_t n = 200000;

    std::string a(n, 'a');
    std::string b(n, 'a');
    b[n - 1] = 'b';

    input_data_ = {a, b};
    expected_ = 1;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_{1};
};

TEST_P(TsarkovKRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TsarkovKLexicographicStringCompareMPI, TsarkovKLexicographicStringCompareSEQ>(
        PPC_SETTINGS_tsarkov_k_lexicographic_string_compare);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TsarkovKRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TsarkovKRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace tsarkov_k_lexicographic_string_compare
