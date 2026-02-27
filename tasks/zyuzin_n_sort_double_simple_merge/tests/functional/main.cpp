#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zyuzin_n_sort_double_simple_merge/common/include/common.hpp"
#include "zyuzin_n_sort_double_simple_merge/mpi/include/ops_mpi.hpp"
#include "zyuzin_n_sort_double_simple_merge/seq/include/ops_seq.hpp"

namespace zyuzin_n_sort_double_simple_merge {

class ZyuzinNSortDoubleSimpleMergeFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_";
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params);
    expected_output_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > 1e-12) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<double> expected_output_;
};

namespace {

TEST_P(ZyuzinNSortDoubleSimpleMergeFuncTests, SortDoubleSimpleMerge) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(1, std::vector<double>{3.5, -2.1, 0.0, 1.1, -3.3, 2.2, -1.4, 5.6},
                    std::vector<double>{-3.3, -2.1, -1.4, 0.0, 1.1, 2.2, 3.5, 5.6}),

    std::make_tuple(2, std::vector<double>{-5.07, -3.3, -1.12, 0.4, 1.111, 3.0, 5.25},
                    std::vector<double>{-5.07, -3.3, -1.12, 0.4, 1.111, 3.0, 5.25}),

    std::make_tuple(3, std::vector<double>{5.0, 3.4, 1.5, 0.0, -1.0, -3.01, -5.0},
                    std::vector<double>{-5.0, -3.01, -1.0, 0.0, 1.5, 3.4, 5.0}),

    std::make_tuple(4, std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0}, std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0}),

    std::make_tuple(5, std::vector<double>{}, std::vector<double>{}),

    std::make_tuple(6, std::vector<double>{42.0}, std::vector<double>{42.0})};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ZyuzinNSortDoubleWithSimpleMergeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_zyuzin_n_sort_double_simple_merge),
                                           ppc::util::AddFuncTask<ZyuzinNSortDoubleWithSimpleMergeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_zyuzin_n_sort_double_simple_merge));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    ZyuzinNSortDoubleSimpleMergeFuncTests::PrintFuncTestName<ZyuzinNSortDoubleSimpleMergeFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, ZyuzinNSortDoubleSimpleMergeFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zyuzin_n_sort_double_simple_merge
