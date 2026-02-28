#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "vinyaikina_e_vert_ribbon_scheme/common/include/common.hpp"
#include "vinyaikina_e_vert_ribbon_scheme/mpi/include/ops_mpi.hpp"
#include "vinyaikina_e_vert_ribbon_scheme/seq/include/ops_seq.hpp"

namespace vinyaikina_e_vert_ribbon_scheme {

class VinyaikinaEVertRibbonSchemeFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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

TEST_P(VinyaikinaEVertRibbonSchemeFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<VinyaikinaEVertRibbonSchemeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_vinyaikina_e_vert_ribbon_scheme),
                                           ppc::util::AddFuncTask<VinyaikinaEVertRibbonSchemeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_vinyaikina_e_vert_ribbon_scheme));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    VinyaikinaEVertRibbonSchemeFuncTests::PrintFuncTestName<VinyaikinaEVertRibbonSchemeFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, VinyaikinaEVertRibbonSchemeFuncTests, kGtestValues, kPerfTestName);

}  // namespace

namespace {

const std::array<TestType, 5> kEdgeParams = {std::make_tuple(1, "one"), std::make_tuple(2, "two"),
                                             std::make_tuple(10, "ten"), std::make_tuple(50, "fifty"),
                                             std::make_tuple(100, "hundred")};

const auto kEdgeTasksList = std::tuple_cat(ppc::util::AddFuncTask<VinyaikinaEVertRibbonSchemeMPI, InType>(
                                               kEdgeParams, PPC_SETTINGS_vinyaikina_e_vert_ribbon_scheme),
                                           ppc::util::AddFuncTask<VinyaikinaEVertRibbonSchemeSEQ, InType>(
                                               kEdgeParams, PPC_SETTINGS_vinyaikina_e_vert_ribbon_scheme));

const auto kEdgeValues = ppc::util::ExpandToValues(kEdgeTasksList);

const auto kEdgeTestName =
    VinyaikinaEVertRibbonSchemeFuncTests::PrintFuncTestName<VinyaikinaEVertRibbonSchemeFuncTests>;

INSTANTIATE_TEST_SUITE_P(EdgeCaseTests, VinyaikinaEVertRibbonSchemeFuncTests, kEdgeValues, kEdgeTestName);

TEST(VinyaikinaEVertRibbonSchemeSeqValidation, RejectsZeroInput) {
  VinyaikinaEVertRibbonSchemeSEQ task(0);
  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(VinyaikinaEVertRibbonSchemeSeqValidation, RejectsNegativeInput) {
  VinyaikinaEVertRibbonSchemeSEQ task(-3);
  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

}  // namespace

}  // namespace vinyaikina_e_vert_ribbon_scheme
