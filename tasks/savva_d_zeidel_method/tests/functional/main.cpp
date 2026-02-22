#include <gtest/gtest.h>
#include <stb/stb_image.h>

// #include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "savva_d_zeidel_method/common/include/common.hpp"
#include "savva_d_zeidel_method/mpi/include/ops_mpi.hpp"
#include "savva_d_zeidel_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace savva_d_zeidel_method {

static std::ostream &operator<<(std::ostream &os, const TestType &test_param);

std::ostream &operator<<(std::ostream &os, const TestType &test_param) {
  const auto &in = test_param.in;

  const auto &name = test_param.name;

  os << "Test[" << name << ", n=" << in.n << ", a.size=" << in.a.size() << ", b.size=" << in.b.size() << "]";
  return os;
}

class SavvaDZeidelFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
  // тест один общий
 public:
  static std::string PrintTestParam(const TestType &test_param) {  // конструктор названия тестов
    const auto &seidelstruct = test_param.in;
    const auto &name_test = test_param.name;
    return name_test + "_" + "matrix_size_" + std::to_string(seidelstruct.n);
  }

 protected:
  void SetUp() override {  // здесь данные готовятся - например читаются изображения
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = params.in;
    right_output_data_ = params.out;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != right_output_data_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (0.001 < std::abs(output_data[i] - right_output_data_[i])) {
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
  OutType right_output_data_;
};

namespace {
// реализация (но пока не запуск) тестов
TEST_P(SavvaDZeidelFuncTests, MatmulFromPic) {  // не изменяется во всех задачах - генерация теста с параметрами
  ExecuteTest(GetParam());
}

const SeidelInput kParam1{.n = 0, .a = {}, .b = {}};
const OutType kVec1{};

const SeidelInput kParam2{.n = 1, .a = {2.0}, .b = {4.0}};
const OutType kVec2{2.0};

const SeidelInput kParam3{.n = 2, .a = {3.0, 1.0, 2.0, 5.0}, .b = {5.0, 12.0}};
const OutType kVec3{1.0, 2.0};

const SeidelInput kParam4{.n = 3, .a = {10.0, -1.0, 2.0, -2.0, 15.0, 3.0, 1.0, 2.0, 20.0}, .b = {11.0, 16.0, 23.0}};
const OutType kVec4{1.0, 1.0, 1.0};

const double kV50 = 353950.0 / 1068421.0;
const double kV51 = 485788.0 / 1068421.0;
const double kV52 = 61566.0 / 1068421.0;
const double kV53 = 503188.0 / 1068421.0;

const SeidelInput kParam5{.n = 4,
                          .a = {5.0, 0.5, 1.2, 0.1, 0.2, 6.0, 0.3, 0.4, 1.1, 0.2, 7.0, 0.3, 0.1, 0.4, 0.3, 8.0},
                          .b = {2.0, 3.0, 1.0, 4.0}};
const OutType kVec5{kV50, kV51, kV52, kV53};

const double kV60 = -3082820798382051.0 / 12480150365419456.0;
const double kV61 = 2481857355488935.0 / 6240075182709728.0;
const double kV62 = -8818834781200853.0 / 64107151648602824.0;

const SeidelInput kParam6{.n = 3,
                          .a = {-8.731, 0.214, -0.517, 0.421, 10.842, -0.318, -0.356, 0.419, -9.953},
                          .b = {2.312948, 4.251926, 1.623761}};
const OutType kVec6{kV60, kV61, kV62};
// std::make_tuple(param1, vec1, "empty_system"),
const std::array<TestType, 6> kTestParam = {
    {{.in = kParam1, .out = kVec1, .name = "empty_system"},
     {.in = kParam2, .out = kVec2, .name = "single_equation"},
     {.in = kParam3, .out = kVec3, .name = "two_by_two_system"},
     {.in = kParam4, .out = kVec4, .name = "three_by_three_negative"},
     {.in = kParam5, .out = kVec5, .name = "four_by_four_fractional"},
     {.in = kParam6, .out = kVec6, .name = "three_by_three_negative_fractional"}}};

// не изменяется (определяет какие тесты будем запускать - сек и мпай )
const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SavvaDZeidelSEQ, InType>(kTestParam, PPC_SETTINGS_savva_d_zeidel_method),
                   ppc::util::AddFuncTask<SavvaDZeidelMPI, InType>(kTestParam, PPC_SETTINGS_savva_d_zeidel_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SavvaDZeidelFuncTests::PrintFuncTestName<SavvaDZeidelFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, SavvaDZeidelFuncTests, kGtestValues,
                         kPerfTestName);  // здесь запуск тестов

}  // namespace

}  // namespace savva_d_zeidel_method
