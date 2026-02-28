#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yusupkina_m_seidel_method/common/include/common.hpp"
#include "yusupkina_m_seidel_method/mpi/include/ops_mpi.hpp"
#include "yusupkina_m_seidel_method/seq/include/ops_seq.hpp"

namespace yusupkina_m_seidel_method {

class YusupkinaMSeidelMethodFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string test_name = std::get<1>(params);
    if (test_name.find("single") != std::string::npos) {
      input_data_ = InType{.matrix = {5.0}, .rhs = {10.0}, .n = 1};
      expected_solution_ = {2.0};
    } else if (test_name.find("identity") != std::string::npos) {
      input_data_ = InType{.matrix = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, .rhs = {3.0, 5.0, 7.0}, .n = 3};
      expected_solution_ = {3.0, 5.0, 7.0};
    } else if (test_name.find("simple_4x4") != std::string::npos) {
      input_data_ =
          InType{.matrix = {10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 10.0},
                 .rhs = {13.0, 13.0, 13.0, 13.0},
                 .n = 4};
      expected_solution_ = {1.0, 1.0, 1.0, 1.0};
    } else if (test_name.find("fractional_matrix") != std::string::npos) {
      input_data_ = InType{.matrix = {3.5, 0.5, 0.5, 4.5}, .rhs = {4.0, 5.0}, .n = 2};
      expected_solution_ = {1.0, 1.0};
    } else if (test_name.find("fractional_solution") != std::string::npos) {
      input_data_ = InType{.matrix = {4.0, 1.0, 1.0, 3.0}, .rhs = {5.0, 5.0}, .n = 2};
      expected_solution_ = {10.0 / 11.0, 15.0 / 11.0};
    } else if (test_name.find("negative") != std::string::npos) {
      input_data_ =
          InType{.matrix = {10.0, -1.0, 2.0, -2.0, 15.0, 3.0, 1.0, 2.0, 20.0}, .rhs = {11.0, 16.0, 23.0}, .n = 3};
      expected_solution_ = {1.0, 1.0, 1.0};
    } else if (test_name.find("negative_solution") != std::string::npos) {
      input_data_ = InType{.matrix = {10.0, 1.0, 1.0, 10.0}, .rhs = {-11.0, -11.0}, .n = 2};
      expected_solution_ = {-1.0, -1.0};
    } else if (test_name.find("large_numbers") != std::string::npos) {
      input_data_ = InType{.matrix = {1000.0, 100.0, 100.0, 1000.0}, .rhs = {1100.0, 1100.0}, .n = 2};
      expected_solution_ = {1.0, 1.0};
    } else if (test_name.find("large_solution") != std::string::npos) {
      input_data_ = InType{.matrix = {10.0, 1.0, 1.0, 10.0}, .rhs = {1100.0, 1100.0}, .n = 2};
      expected_solution_ = {100.0, 100.0};
    }

    else if (test_name.find("almost_singular") != std::string::npos) {
      input_data_ = InType{.matrix = {10.0, 9.9, 9.9, 10.0}, .rhs = {19.9, 19.9}, .n = 2};
      expected_solution_ = {1.0, 1.0};
    } else if (test_name.find("sparse") != std::string::npos) {
      input_data_ = InType{.matrix = {10.0, 1.0, 0.0, 0.0, 0.0, 1.0,  10.0, 1.0, 0.0, 0.0, 0.0, 1.0, 10.0,
                                      1.0,  0.0, 0.0, 0.0, 1.0, 10.0, 1.0,  0.0, 0.0, 0.0, 1.0, 10.0},
                           .rhs = {11.0, 12.0, 12.0, 12.0, 11.0},
                           .n = 5};
      expected_solution_ = {1.0, 1.0, 1.0, 1.0, 1.0};
    } else if (test_name.find("non_ones") != std::string::npos) {
      input_data_ =
          InType{.matrix = {10.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 10.0}, .rhs = {15.0, 24.0, 33.0}, .n = 3};
      expected_solution_ = {1.0, 2.0, 3.0};
    } else if (test_name.find("mixed_sign") != std::string::npos) {
      input_data_ =
          InType{.matrix = {10.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 10.0}, .rhs = {10.0, -8.0, 10.0}, .n = 3};
      expected_solution_ = {1.0, -1.0, 1.0};
    } else if (test_name.find("zero_solution") != std::string::npos) {
      input_data_ = InType{.matrix = {10.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 10.0}, .rhs = {0.0, 0.0, 0.0}, .n = 3};
      expected_solution_ = {0.0, 0.0, 0.0};
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double epsilon = 1e-5;

    if (output_data.size() != expected_solution_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_solution_[i]) > epsilon) {
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
  OutType expected_solution_;
};

namespace {

TEST_P(YusupkinaMSeidelMethodFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 14> kTestParam = {std::make_tuple(0, "single"),
                                             std::make_tuple(1, "identity"),
                                             std::make_tuple(2, "simple_4x4"),
                                             std::make_tuple(3, "fractional_matrix"),
                                             std::make_tuple(4, "fractional_solution"),
                                             std::make_tuple(5, "negative"),
                                             std::make_tuple(6, "negative_solution"),
                                             std::make_tuple(7, "large_numbers"),
                                             std::make_tuple(8, "large_solution"),
                                             std::make_tuple(9, "almost_singular"),
                                             std::make_tuple(10, "sparse"),
                                             std::make_tuple(11, "non_ones"),
                                             std::make_tuple(12, "mixed_sign"),
                                             std::make_tuple(13, "zero_solution")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<YusupkinaMSeidelMethodMPI, InType>(kTestParam, PPC_SETTINGS_yusupkina_m_seidel_method),
    ppc::util::AddFuncTask<YusupkinaMSeidelMethodSEQ, InType>(kTestParam, PPC_SETTINGS_yusupkina_m_seidel_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = YusupkinaMSeidelMethodFuncTests::PrintFuncTestName<YusupkinaMSeidelMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, YusupkinaMSeidelMethodFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace yusupkina_m_seidel_method
