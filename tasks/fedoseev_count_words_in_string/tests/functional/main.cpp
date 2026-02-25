#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <cctype>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fedoseev_count_words_in_string/common/include/common.hpp"
#include "fedoseev_count_words_in_string/mpi/include/ops_mpi.hpp"
#include "fedoseev_count_words_in_string/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace fedoseev_count_words_in_string {

using InType = fedoseev_count_words_in_string::InType;
using OutType = fedoseev_count_words_in_string::OutType;
using TestType = fedoseev_count_words_in_string::TestType;

class FedoseevRunFuncTestsWordsCount : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto& input = std::get<0>(test_param);
    const auto& expected = std::get<1>(test_param);
    std::string safe_name;
    
    if (input.empty()) {
      safe_name = "empty_string";
    } else if (input == "   ") {
      safe_name = "only_spaces";
    } else if (input == "\t\n ") {
      safe_name = "whitespace_chars";
    } else if (input == "hello") {
      safe_name = "single_word";
    } else if (input == "hello world") {
      safe_name = "two_words";
    } else if (input == "one  two   three") {
      safe_name = "multiple_spaces";
    } else if (input == "  leading and trailing  ") {
      safe_name = "leading_trailing_spaces";
    } else if (input == "newline\nseparated\nwords") {
      safe_name = "newline_separated";
    } else if (input == "\t tabs\tand  spaces \n mix") {
      safe_name = "mixed_whitespace";
    } else if (input == "multi\nline with \t mixed\t\n whitespace") {
      safe_name = "complex_whitespace";
    } else if (input == "punctuation,shouldn't-break!words?") {
      safe_name = "punctuation";
    } else if (input == "C++ programming is fun!") {
      safe_name = "with_symbols";
    } else if (input == "русский  текст  да") {
      safe_name = "cyrillic_text";
    } else if (input == "emoji 👍🏽rocks") {
      safe_name = "emoji";
    } else {
      safe_name = "test_" + std::to_string(expected);
    }
    
    return safe_name + "_expected_" + std::to_string(expected);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(FedoseevRunFuncTestsWordsCount, CountWordsTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 14> kTestParam = {

    std::make_tuple(std::string{""}, 0),
    std::make_tuple(std::string{"   "}, 0),
    std::make_tuple(std::string{"\t\n "}, 0),
    std::make_tuple(std::string{"hello"}, 1),
    std::make_tuple(std::string{"hello world"}, 2),
    std::make_tuple(std::string{"one  two   three"}, 3),
    std::make_tuple(std::string{"  leading and trailing  "}, 3),
    std::make_tuple(std::string{"newline\nseparated\nwords"}, 3),
    std::make_tuple(std::string{"\t tabs\tand  spaces \n mix"}, 4),
    std::make_tuple(std::string{"multi\nline with \t mixed\t\n whitespace"}, 5),
    std::make_tuple(std::string{"punctuation,shouldn't-break!words?"}, 1),
    std::make_tuple(std::string{"C++ programming is fun!"}, 4),
    std::make_tuple(std::string{"русский  текст  да"}, 3),
    std::make_tuple(std::string{"emoji 👍🏽rocks"}, 2)
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<FedoseevCountWordsInStringMPI, InType>(
                                               kTestParam, PPC_SETTINGS_fedoseev_count_words_in_string),
                                           ppc::util::AddFuncTask<FedoseevCountWordsInStringSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_fedoseev_count_words_in_string));
const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kFuncTestName = FedoseevRunFuncTestsWordsCount::PrintFuncTestName<FedoseevRunFuncTestsWordsCount>;

INSTANTIATE_TEST_SUITE_P(WordsCountTests, FedoseevRunFuncTestsWordsCount, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace fedoseev_count_words_in_string