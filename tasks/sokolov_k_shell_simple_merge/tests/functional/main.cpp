#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "sokolov_k_shell_simple_merge/common/include/common.hpp"
#include "sokolov_k_shell_simple_merge/mpi/include/ops_mpi.hpp"
#include "sokolov_k_shell_simple_merge/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sokolov_k_shell_simple_merge {

class SokolovKShellSimpleMergeFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(SokolovKShellSimpleMergeFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SokolovKShellSimpleMergeMPI, InType>(kTestParam, PPC_SETTINGS_sokolov_k_shell_simple_merge),
    ppc::util::AddFuncTask<SokolovKShellSimpleMergeSEQ, InType>(kTestParam, PPC_SETTINGS_sokolov_k_shell_simple_merge));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SokolovKShellSimpleMergeFuncTests::PrintFuncTestName<SokolovKShellSimpleMergeFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, SokolovKShellSimpleMergeFuncTests, kGtestValues, kPerfTestName);

const std::array<TestType, 8> kEdgeCaseParams = {
    std::make_tuple(1, "single_element"),     std::make_tuple(2, "two_elements"),
    std::make_tuple(4, "four_elements"),      std::make_tuple(10, "ten_elements"),
    std::make_tuple(25, "twentyfive"),        std::make_tuple(50, "fifty_elements"),
    std::make_tuple(100, "hundred_elements"), std::make_tuple(500, "five_hundred")};

const auto kEdgeTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SokolovKShellSimpleMergeMPI, InType>(
                                                   kEdgeCaseParams, PPC_SETTINGS_sokolov_k_shell_simple_merge),
                                               ppc::util::AddFuncTask<SokolovKShellSimpleMergeSEQ, InType>(
                                                   kEdgeCaseParams, PPC_SETTINGS_sokolov_k_shell_simple_merge));

const auto kEdgeGtestValues = ppc::util::ExpandToValues(kEdgeTestTasksList);

INSTANTIATE_TEST_SUITE_P(EdgeCaseTests, SokolovKShellSimpleMergeFuncTests, kEdgeGtestValues, kPerfTestName);

}  // namespace

}  // namespace sokolov_k_shell_simple_merge
