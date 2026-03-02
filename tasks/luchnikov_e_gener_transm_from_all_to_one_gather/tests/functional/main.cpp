#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

class LuchnikovEGenerTransformFromAllToOneGatherFuncTestsSmallValues
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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
    return (input_data_ > 0) && (output_data > 0);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

class LuchnikovEGenerTransformFromAllToOneGatherFuncTestsMediumValues
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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
    return (input_data_ > 0) && (output_data > 0);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

class LuchnikovEGenerTransformFromAllToOneGatherFuncTestsLargeValues
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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
    return (input_data_ > 0) && (output_data > 0);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(LuchnikovEGenerTransformFromAllToOneGatherFuncTestsSmallValues, SmallInputValues) {
  ExecuteTest(GetParam());
}

TEST_P(LuchnikovEGenerTransformFromAllToOneGatherFuncTestsMediumValues, MediumInputValues) {
  ExecuteTest(GetParam());
}

TEST_P(LuchnikovEGenerTransformFromAllToOneGatherFuncTestsLargeValues, LargeInputValues) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParamSmall = {std::make_tuple(1, "1"), std::make_tuple(2, "2"),
                                                 std::make_tuple(3, "3"), std::make_tuple(4, "4"),
                                                 std::make_tuple(5, "5")};

const std::array<TestType, 5> kTestParamMedium = {std::make_tuple(10, "10"), std::make_tuple(15, "15"),
                                                  std::make_tuple(20, "20"), std::make_tuple(25, "25"),
                                                  std::make_tuple(30, "30")};

const std::array<TestType, 5> kTestParamLarge = {std::make_tuple(50, "50"), std::make_tuple(75, "75"),
                                                 std::make_tuple(100, "100"), std::make_tuple(125, "125"),
                                                 std::make_tuple(150, "150")};

const auto kTestTasksListSmall =
    std::tuple_cat(ppc::util::AddFuncTask<LuchnikovEGenerTransformFromAllToOneGatherMPI, InType>(
                       kTestParamSmall, PPC_SETTINGS_luchnikov_e_gener_transm_from_all_to_one_gather),
                   ppc::util::AddFuncTask<LuchnikovEGenerTransformFromAllToOneGatherSEQ, InType>(
                       kTestParamSmall, PPC_SETTINGS_luchnikov_e_gener_transm_from_all_to_one_gather));

const auto kTestTasksListMedium =
    std::tuple_cat(ppc::util::AddFuncTask<LuchnikovEGenerTransformFromAllToOneGatherMPI, InType>(
                       kTestParamMedium, PPC_SETTINGS_luchnikov_e_gener_transm_from_all_to_one_gather),
                   ppc::util::AddFuncTask<LuchnikovEGenerTransformFromAllToOneGatherSEQ, InType>(
                       kTestParamMedium, PPC_SETTINGS_luchnikov_e_gener_transm_from_all_to_one_gather));

const auto kTestTasksListLarge =
    std::tuple_cat(ppc::util::AddFuncTask<LuchnikovEGenerTransformFromAllToOneGatherMPI, InType>(
                       kTestParamLarge, PPC_SETTINGS_luchnikov_e_gener_transm_from_all_to_one_gather),
                   ppc::util::AddFuncTask<LuchnikovEGenerTransformFromAllToOneGatherSEQ, InType>(
                       kTestParamLarge, PPC_SETTINGS_luchnikov_e_gener_transm_from_all_to_one_gather));

const auto kGtestValuesSmall = ppc::util::ExpandToValues(kTestTasksListSmall);
const auto kGtestValuesMedium = ppc::util::ExpandToValues(kTestTasksListMedium);
const auto kGtestValuesLarge = ppc::util::ExpandToValues(kTestTasksListLarge);

const auto kPerfTestNameSmall = LuchnikovEGenerTransformFromAllToOneGatherFuncTestsSmallValues::PrintFuncTestName<
    LuchnikovEGenerTransformFromAllToOneGatherFuncTestsSmallValues>;

const auto kPerfTestNameMedium = LuchnikovEGenerTransformFromAllToOneGatherFuncTestsMediumValues::PrintFuncTestName<
    LuchnikovEGenerTransformFromAllToOneGatherFuncTestsMediumValues>;

const auto kPerfTestNameLarge = LuchnikovEGenerTransformFromAllToOneGatherFuncTestsLargeValues::PrintFuncTestName<
    LuchnikovEGenerTransformFromAllToOneGatherFuncTestsLargeValues>;

INSTANTIATE_TEST_SUITE_P(SmallValueTests, LuchnikovEGenerTransformFromAllToOneGatherFuncTestsSmallValues,
                         kGtestValuesSmall, kPerfTestNameSmall);

INSTANTIATE_TEST_SUITE_P(MediumValueTests, LuchnikovEGenerTransformFromAllToOneGatherFuncTestsMediumValues,
                         kGtestValuesMedium, kPerfTestNameMedium);

INSTANTIATE_TEST_SUITE_P(LargeValueTests, LuchnikovEGenerTransformFromAllToOneGatherFuncTestsLargeValues,
                         kGtestValuesLarge, kPerfTestNameLarge);

}  // namespace

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
