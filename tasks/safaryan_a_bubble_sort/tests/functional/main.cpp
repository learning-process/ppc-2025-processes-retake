#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>

#include "safaryan_a_bubble_sort/common/include/common.hpp"
#include "safaryan_a_bubble_sort/mpi/include/ops_mpi.hpp"
#include "safaryan_a_bubble_sort/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace safaryan_a_bubble_sort {
class SafaryanABubbleSortRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    size_t dot = test_param.find('.');
    return test_param.substr(0, dot);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_safaryan_a_bubble_sort, params);
    std::ifstream file(path);

    int value = 0;
    while (file >> value) {
      in_.push_back(value);
    }

    file.close();
  }
  bool CheckTestOutputData(OutType &output_data) final {
    return std::ranges::is_sorted(output_data);
  }

  InType GetTestInputData() final {
    return in_;
  }

 private:
  InType in_;
};

namespace {
TEST_P(SafaryanABubbleSortRunFuncTestsProcesses, BubbleSortFromFiles) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {"test1.txt", "test2.txt", "test3.txt"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SafaryanABubbleSortMPI, InType>(kTestParam, PPC_SETTINGS_safaryan_a_bubble_sort),
    ppc::util::AddFuncTask<SafaryanABubbleSortSEQ, InType>(kTestParam, PPC_SETTINGS_safaryan_a_bubble_sort));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    SafaryanABubbleSortRunFuncTestsProcesses::PrintFuncTestName<SafaryanABubbleSortRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(BubbleSort, SafaryanABubbleSortRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace safaryan_a_bubble_sort
