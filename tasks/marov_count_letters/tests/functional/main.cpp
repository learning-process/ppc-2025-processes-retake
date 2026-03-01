#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include "marov_count_letters/common/include/common.hpp"
#include "marov_count_letters/mpi/include/ops_mpi.hpp"
#include "marov_count_letters/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace marov_count_letters {

class MarovCountLettersFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<ParamType>& info) {
    return "test_" + std::to_string(info.index);
  }

 protected:
  void SetUp() override {
    const auto& [input_str, expected] = std::get<2>(GetParam());
    input_data_ = input_str;
    expected_output_ = expected;
  }

  bool CheckTestOutputData(OutType& result) final {
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

TEST_P(MarovCountLettersFuncTests, CountLetterChars) {
  ExecuteTest(GetParam());
}

const std::array kTestCases = {
    std::make_tuple(std::string("Hello, World!"), 10),
    std::make_tuple(std::string("12345"), 0),
    std::make_tuple(std::string(""), 0),
    std::make_tuple(std::string("abcdef"), 6),
    std::make_tuple(std::string("ABC xyz"), 7),
    std::make_tuple(std::string("Test123String"), 10),
    std::make_tuple(std::string("!!!"), 0),
    std::make_tuple(std::string("aBcDeFgHiJkLmNoPqRsTuVwXyZ"), 26),
};

const auto kAllTestTasks = std::tuple_cat(
    ppc::util::AddFuncTask<MarovCountLettersMPI, InType>(kTestCases, PPC_SETTINGS_marov_count_letters),
    ppc::util::AddFuncTask<MarovCountLettersSEQ, InType>(kTestCases, PPC_SETTINGS_marov_count_letters));

INSTANTIATE_TEST_SUITE_P(LetterCountFuncTests, MarovCountLettersFuncTests,
                         ppc::util::ExpandToValues(kAllTestTasks),
                         MarovCountLettersFuncTests::PrintTestParam);

}  // namespace
}  // namespace marov_count_letters
