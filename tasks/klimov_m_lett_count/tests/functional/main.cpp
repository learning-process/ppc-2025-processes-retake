#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <random>
#include <string>
#include <tuple>

#include "klimov_m_lett_count/common/include/common.hpp"
#include "klimov_m_lett_count/mpi/include/ops_mpi.hpp"
#include "klimov_m_lett_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace klimov_m_lett_count {

class KlimovMLettCountTest : public ppc::util::BaseRunFuncTests<InputType, OutputType, TestParam> {
 public:
  static std::string PrintTestParam(const TestParam &param) {
    return std::get<1>(param);
  }

 protected:
  void SetUp() override {
    auto full_param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    auto inner = std::get<0>(full_param);
    std::string pattern = std::get<0>(inner);
    int expected = std::get<1>(inner);
    if (pattern != "generate") {
      input_data_ = pattern;
      expected_result_ = expected;
    } else {
      input_data_ = GenerateRandomString(expected);
      expected_result_ = expected;
    }
  }

  bool CheckTestOutputData(OutputType &output) final {
    return expected_result_ == output;
  }

  InputType GetTestInputData() final {
    return input_data_;
  }

 private:
  InputType input_data_;
  OutputType expected_result_ = 0;

  static std::string GenerateRandomString(size_t letter_count) {
    std::mt19937 rng(static_cast<unsigned int>(letter_count));
    std::uniform_int_distribution<size_t> len_dist(100, 500);
    size_t total_len = len_dist(rng);
    std::string result;
    result.reserve(total_len);

    std::string letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::uniform_int_distribution<size_t> letter_idx(0, letters.size() - 1);
    for (size_t i = 0; i < letter_count; ++i) {
      result += letters[letter_idx(rng)];
    }
    std::uniform_int_distribution<> digit_dist(48, 57);
    for (size_t i = letter_count; i < total_len; ++i) {
      result += static_cast<char>(digit_dist(rng));
    }
    std::shuffle(result.begin(), result.end(), rng);
    return result;
  }
};

namespace {

TEST_P(KlimovMLettCountTest, CountLetters) {
  ExecuteTest(GetParam());
}

const std::array<TestParam, 20> kTestCases = {
    {std::make_tuple(std::make_tuple("", 0), "empty"),
     std::make_tuple(std::make_tuple("abcd", 4), "only_letters"),
     std::make_tuple(std::make_tuple("aabcd123abcd123abcd", 13), "mixed1"),
     std::make_tuple(std::make_tuple("abcd_____________123abcd", 8), "mixed2"),
     std::make_tuple(std::make_tuple("a", 1), "single_letter"),
     std::make_tuple(std::make_tuple("126756", 0), "only_digits"),
     std::make_tuple(std::make_tuple("a1a1a1a1a1a1a1a1a1a1a1a1", 12), "alternating"),
     std::make_tuple(std::make_tuple("!@467678&*()", 0), "punctuation"),
     std::make_tuple(std::make_tuple("aaaaaaaaaaaaaaaaaaaa", 20), "many_letters"),
     std::make_tuple(std::make_tuple("tatatatatatatatatatatatatatatatatatatatata", 42), "pattern"),
     std::make_tuple(std::make_tuple("er11er11er11er11", 8), "digits_in_between"),
     std::make_tuple(std::make_tuple("eee___eee__", 6), "underscores"),
     std::make_tuple(std::make_tuple("eee___eee__EEE", 9), "uppercase"),
     std::make_tuple(std::make_tuple("EEEE___EEEE", 8), "all_uppercase"),
     std::make_tuple(
         std::make_tuple(
             "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
             100),
         "long_string"),
     std::make_tuple(std::make_tuple("вопылкнгш", 0), "cyrillic"),
     std::make_tuple(std::make_tuple("aa", 2), "two_letters"),
     std::make_tuple(std::make_tuple("aaa", 3), "three_letters"),
     std::make_tuple(std::make_tuple("aabb0123456767", 4), "letters_and_digits"),
     std::make_tuple(std::make_tuple("generate", 100), "generated")}};

const auto kTaskList = std::tuple_cat(
    ppc::util::AddFuncTask<KlimovMLettCountMPI, InputType>(kTestCases, "tasks/klimov_m_lett_count/settings.json"),
    ppc::util::AddFuncTask<KlimovMLettCountSEQ, InputType>(kTestCases, "tasks/klimov_m_lett_count/settings.json"));

const auto kGtestValues = ppc::util::ExpandToValues(kTaskList);

const auto kTestName = KlimovMLettCountTest::PrintFuncTestName<KlimovMLettCountTest>;

INSTANTIATE_TEST_SUITE_P(KlimovLettCountSuite, KlimovMLettCountTest, kGtestValues, kTestName);

}  // namespace

}  // namespace klimov_m_lett_count
