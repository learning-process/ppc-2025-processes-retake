#include <gtest/gtest.h>
#include <array>
#include <string>
#include <tuple>
#include <vector>

#include "marov_radix_sort_double/common/include/common.hpp"
#include "marov_radix_sort_double/mpi/include/ops_mpi.hpp"
#include "marov_radix_sort_double/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace marov_radix_sort_double {

class MarovRadixSortDoubleFuncTests
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(
      const testing::TestParamInfo<ParamType>& info) {
    return "test_" + std::to_string(info.index);
  }

 protected:
  void SetUp() override {
    const auto& [input_vec, expected_vec] = std::get<2>(GetParam());
    input_data_ = input_vec;
    expected_output_ = expected_vec;
  }

  bool CheckTestOutputData(OutType& result) final {
    return result == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(MarovRadixSortDoubleFuncTests, RadixSortDoubleSimpleMerge) {
  ExecuteTest(GetParam());
}

const std::array kTestCases = {
    std::make_tuple(std::vector<double>{5.0, 2.0, 8.0, 1.0, 9.0},
                    std::vector<double>{1.0, 2.0, 5.0, 8.0, 9.0}),
    std::make_tuple(std::vector<double>{}, std::vector<double>{}),
    std::make_tuple(std::vector<double>{1.0}, std::vector<double>{1.0}),
    std::make_tuple(std::vector<double>{3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0},
                    std::vector<double>{1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0}),
    std::make_tuple(std::vector<double>{-5.0, 3.0, -1.0, 0.0, 2.0},
                    std::vector<double>{-5.0, -1.0, 0.0, 2.0, 3.0}),
    std::make_tuple(std::vector<double>{1.5, 3.7, 2.1, 0.5, 4.9},
                    std::vector<double>{0.5, 1.5, 2.1, 3.7, 4.9}),
};

const auto kAllTestTasks =
    std::tuple_cat(ppc::util::AddFuncTask<MarovRadixSortDoubleMpi, InType>(
                       kTestCases, PPC_SETTINGS_marov_radix_sort_double),
                   ppc::util::AddFuncTask<MarovRadixSortDoubleSeq, InType>(
                       kTestCases, PPC_SETTINGS_marov_radix_sort_double));

INSTANTIATE_TEST_SUITE_P(
    RadixSortDoubleFuncTests, MarovRadixSortDoubleFuncTests,
    ppc::util::ExpandToValues(kAllTestTasks),
    MarovRadixSortDoubleFuncTests::PrintTestParam);

}  // namespace
}  // namespace marov_radix_sort_double
