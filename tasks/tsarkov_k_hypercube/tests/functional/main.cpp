#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "tsarkov_k_hypercube/common/include/common.hpp"
#include "tsarkov_k_hypercube/mpi/include/ops_mpi.hpp"
#include "tsarkov_k_hypercube/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tsarkov_k_hypercube {

class TsarkovKHypercubeFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType tt = std::get<static_cast<std::uint8_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_.resize(3);
    input_data_[0] = std::get<0>(tt)[0];
    input_data_[1] = std::get<0>(tt)[1];
    input_data_[2] = std::get<0>(tt)[2];
  }

  bool CheckTestOutputData(OutType &output_data) final {
    TestType tt = std::get<static_cast<std::uint8_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    return output_data == std::get<0>(tt)[2];
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(TsarkovKHypercubeFuncTests, HypercubeRoute) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {std::make_tuple(std::vector<int>{0, 3, 4}, "far"),
                                            std::make_tuple(std::vector<int>{0, 3, 0}, "far_empty"),
                                            std::make_tuple(std::vector<int>{0, 1, 4}, "neighbors"),
                                            std::make_tuple(std::vector<int>{2, 3, 4}, "zero_not_source"),
                                            std::make_tuple(std::vector<int>{2, 2, 4}, "to_itself"),
                                            std::make_tuple(std::vector<int>{2, 1, 4}, "reverse_path"),
                                            std::make_tuple(std::vector<int>{3, 0, 4}, "reverse")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<TsarkovKHypercubeMPI, InType>(kTestParam, PPC_SETTINGS_tsarkov_k_hypercube),
                   ppc::util::AddFuncTask<TsarkovKHypercubeSEQ, InType>(kTestParam, PPC_SETTINGS_tsarkov_k_hypercube));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TsarkovKHypercubeFuncTests::PrintFuncTestName<TsarkovKHypercubeFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TsarkovKHypercubeFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace tsarkov_k_hypercube
