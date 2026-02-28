#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "salena_s_vec_min_val/common/include/common.hpp"
#include "salena_s_vec_min_val/mpi/include/ops_mpi.hpp"
#include "salena_s_vec_min_val/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace salena_s_vec_min_val {

using TestType = std::tuple<int, std::string>;

class VectorMinFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);
    input_data_.resize(size);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    for (int i = 0; i < size; ++i) {
      input_data_[i] = dist(gen);
    }
    if (size > 0) input_data_[size / 2] = -20000;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int expected_min = *std::min_element(input_data_.begin(), input_data_.end());
    return (expected_min == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(VectorMinFuncTests, FindMin) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(10, "10"),
    std::make_tuple(100, "100"),
    std::make_tuple(1005, "1005")
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<TestTaskMPI, InType>(kTestParam, PPC_SETTINGS_salena_s_vec_min_val),
                   ppc::util::AddFuncTask<TestTaskSEQ, InType>(kTestParam, PPC_SETTINGS_salena_s_vec_min_val));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = VectorMinFuncTests::PrintFuncTestName<VectorMinFuncTests>;
INSTANTIATE_TEST_SUITE_P(VecMinTests, VectorMinFuncTests, kGtestValues, kPerfTestName);

}  // namespace salena_s_vec_min_val
