#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "dergynov_s_hypercube/common/include/common.hpp"
#include "dergynov_s_hypercube/mpi/include/ops_mpi.hpp"
#include "dergynov_s_hypercube/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dergynov_s_hypercube {

class DergynovSHypercubeFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto tt = std::get<static_cast<uint8_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_ = std::get<0>(tt);
  }

  bool CheckTestOutputData(OutType &out) final {
    return out == input_[2];
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
};

namespace {

TEST_P(DergynovSHypercubeFuncTests, Run) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTests = {
    std::make_tuple(std::vector<int>{0, 3, 5}, "far"),     std::make_tuple(std::vector<int>{0, 1, 5}, "neighbor"),
    std::make_tuple(std::vector<int>{2, 3, 5}, "mid"),     std::make_tuple(std::vector<int>{1, 1, 5}, "self"),
    std::make_tuple(std::vector<int>{3, 0, 5}, "reverse"), std::make_tuple(std::vector<int>{0, 3, 0}, "empty"),
};

const auto kTasks =
    std::tuple_cat(ppc::util::AddFuncTask<DergynovSHypercubeMPI, InType>(kTests, PPC_SETTINGS_dergynov_s_hypercube),
                   ppc::util::AddFuncTask<DergynovSHypercubeSEQ, InType>(kTests, PPC_SETTINGS_dergynov_s_hypercube));

const auto kValues = ppc::util::ExpandToValues(kTasks);

INSTANTIATE_TEST_SUITE_P(HypercubeTests, DergynovSHypercubeFuncTests, kValues,
                         DergynovSHypercubeFuncTests::PrintFuncTestName<DergynovSHypercubeFuncTests>);

}  // namespace

}  // namespace dergynov_s_hypercube