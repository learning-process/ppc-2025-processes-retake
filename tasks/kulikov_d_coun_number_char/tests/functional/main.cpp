#include <gtest/gtest.h>

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "kulikov_d_coun_number_char/common/include/common.hpp"
#include "kulikov_d_coun_number_char/mpi/include/ops_mpi.hpp"
#include "kulikov_d_coun_number_char/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kulikov_d_coun_number_char {
class KulikovDiffCountNumberCharFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<ParamType> &info) {
    const auto &[filename, expected] = std::get<2>(info.param);
    std::string clean_name = filename;
    if (auto pos = clean_name.find_last_of('.'); pos != std::string::npos) {
      clean_name = clean_name.substr(0, pos);
    }
    return clean_name + "_expected" + std::to_string(expected) + "_idx" + std::to_string(info.index);
  }

 protected:
  void SetUp() override {
    const auto &[filename, expected] = std::get<2>(GetParam());
    expected_output_ = expected;

    std::string full_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_kulikov_d_coun_number_char, filename);

    std::ifstream input_file(full_path);
    if (!input_file) {
      throw std::runtime_error("Test file not found: " + full_path);
    }

    std::string first_line;
    std::string second_line;
    if (!std::getline(input_file, first_line)) {
      throw std::runtime_error("Empty or unreadable first line in: " + filename);
    }
    if (!std::getline(input_file, second_line)) {
      throw std::runtime_error("Missing second line in: " + filename);
    }

    auto trim_cr = [](std::string &s) {
      if (!s.empty() && s.back() == '\r') {
        s.pop_back();
      }
    };
    trim_cr(first_line);
    trim_cr(second_line);

    std::string garbage;
    if (std::getline(input_file, garbage) && !garbage.empty()) {
      throw std::runtime_error("Extra content found in test file: " + filename);
    }

    input_data_ = {std::move(first_line), std::move(second_line)};
  }

  bool CheckTestOutputData(OutType &result) final {
    return result == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_ = 0;
};

namespace {

TEST_P(KulikovDiffCountNumberCharFuncTests, CountDifferingCharacters) {
  ExecuteTest(GetParam());
}

const std::array kTestCases = {std::make_pair("empty_strings.txt", 0),
                               std::make_pair("test_identical.txt", 0),
                               std::make_pair("test_single_diff.txt", 1),
                               std::make_pair("test_empty_first.txt", 3),
                               std::make_pair("test_empty_second.txt", 3),
                               std::make_pair("test_diff_length.txt", 3),
                               std::make_pair("all_different_short.txt", 6),
                               std::make_pair("with_spaces_and_tabs.txt", 2),
                               std::make_pair("case_matters.txt", 1),
                               std::make_pair("unicode_safe_ascii.txt", 0),
                               std::make_pair("long_strings_one_diff_at_end.txt", 1),
                               std::make_pair("special_symbols_mismatch.txt", 2)};

const auto kAllTestTasks = std::tuple_cat(
    ppc::util::AddFuncTask<KulikovDiffCountNumberCharMPI, InType>(kTestCases, PPC_SETTINGS_kulikov_d_coun_number_char),
    ppc::util::AddFuncTask<KulikovDiffCountNumberCharSEQ, InType>(kTestCases, PPC_SETTINGS_kulikov_d_coun_number_char));

INSTANTIATE_TEST_SUITE_P(CharacterDiffFuncTests, KulikovDiffCountNumberCharFuncTests,
                         ppc::util::ExpandToValues(kAllTestTasks), KulikovDiffCountNumberCharFuncTests::PrintTestParam);

}  // namespace
}  // namespace kulikov_d_coun_number_char
