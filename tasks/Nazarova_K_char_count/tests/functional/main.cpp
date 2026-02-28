#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "Nazarova_K_char_count/common/include/common.hpp"
#include "Nazarova_K_char_count/mpi/include/ops_mpi.hpp"
#include "Nazarova_K_char_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nazarova_k_char_count_processes {

class NazarovaKCharCountRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const int n = std::get<0>(test_param);
    const char target = std::get<1>(test_param);
    const bool has_target = std::get<2>(test_param);
    return std::to_string(n) + "_" + std::string(1, target) + "_" + (has_target ? "has" : "no");
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int n = std::get<0>(params);
    const char target = std::get<1>(params);
    const bool has_target = std::get<2>(params);

    input_data_.target = target;
    input_data_.text.assign(static_cast<std::size_t>(std::max(0, n)), 'q');

    expected_ = 0;
    if (has_target) {
      // Put target at every 7th position (including 0) for deterministic expected result.
      for (std::size_t i = 0; i < input_data_.text.size(); i++) {
        if (i % 7 == 0) {
          input_data_.text[i] = target;
          expected_++;
        }
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_ = 0;
};

namespace {

TEST_P(NazarovaKCharCountRunFuncTests, CountChar) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(0, 'a', true), std::make_tuple(100, 'x', false),
                                            std::make_tuple(10000, 'z', true)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<NazarovaKCharCountMPI, InType>(kTestParam, PPC_SETTINGS_Nazarova_K_char_count),
    ppc::util::AddFuncTask<NazarovaKCharCountSEQ, InType>(kTestParam, PPC_SETTINGS_Nazarova_K_char_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = NazarovaKCharCountRunFuncTests::PrintFuncTestName<NazarovaKCharCountRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(CharCountTests, NazarovaKCharCountRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace nazarova_k_char_count_processes
