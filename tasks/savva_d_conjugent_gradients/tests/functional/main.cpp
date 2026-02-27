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

#include "savva_d_conjugent_gradients/common/include/common.hpp"
#include "savva_d_conjugent_gradients/mpi/include/ops_mpi.hpp"
#include "savva_d_conjugent_gradients/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace savva_d_conjugent_gradients {

static std::ostream &operator<<(std::ostream &os, const TestType &test_param);

std::ostream &operator<<(std::ostream &os, const TestType &test_param) {
  const auto &in = test_param.in;

  const auto &name = test_param.name;

  os << "Test[" << name << ", n=" << in.n << ", a.size=" << in.a.size() << ", b.size=" << in.b.size() << "]";
  return os;
}

class SavvaDConjugentGradientsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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
TEST_P(SavvaDConjugentGradientsFuncTests,
       MatmulFromPic) {  // не изменяется во всех задачах - генерация теста с параметрами
  ExecuteTest(GetParam());
}

const InputSystem kParam1{.n = 0, .a = {}, .b = {}};
const OutType kVec1{};

const InputSystem kParam2{.n = 1, .a = {2.0}, .b = {4.0}};
const OutType kVec2{2.0};

const InputSystem kParam3{.n = 2, .a = {7.0, 2.0, 2.0, 5.0}, .b = {11.0, 12.0}};
const OutType kVec3{1.0, 2.0};

const InputSystem kParam4{.n = 3, .a = {3.0, 1.0, 0.0, 1.0, 4.0, 2.0, 0.0, 2.0, 5.0}, .b = {4.0, 7.0, 7.0}};
const OutType kVec4{1.0, 1.0, 1.0};

const double kV50 = 325581.0 / 17119448.0;
const double kV51 = 925413.0 / 17119448.0;
const double kV52 = 353133.0 / 17119448.0;
const double kV53 = 2896595.0 / 17119448.0;

const InputSystem kParam5{
    .n = 4,
    .a = {99.0, -6.0, 5.0, 2.0, -6.0, 50.0, 11.0, 7.0, 5.0, 11.0, 112.0, 0.0, 2.0, 7.0, 0.0, 33.0},
    .b = {2.0, 4.0, 3.0, 6.0}};
const OutType kVec5{kV50, kV51, kV52, kV53};

const double kV60 = 200359.0 / 3581592.0;
const double kV61 = 21711.0 / 298466.0;
const double kV62 = 110773.0 / 596932.0;

const InputSystem kParam6{.n = 3, .a = {15.6, 0.0, -1.2, 0.0, 17.9, -5.4, -1.2, -5.4, 40.2}, .b = {0.65, 0.3, 7.0}};
const OutType kVec6{kV60, kV61, kV62};

const InputSystem kParam7{.n = 3, .a = {3.0, 1.0, 0.0, 1.0, 4.0, 2.0, 0.0, 2.0, 5.0}, .b = {0.0, 0.0, 0.0}};
const OutType kVec7{0.0, 0.0, 0.0};

const std::array<TestType, 7> kTestParam = {{{.in = kParam1, .out = kVec1, .name = "empty_system"},
                                             {.in = kParam2, .out = kVec2, .name = "single_equation"},
                                             {.in = kParam3, .out = kVec3, .name = "two_by_two_system"},
                                             {.in = kParam4, .out = kVec4, .name = "three_by_three_system"},
                                             {.in = kParam5, .out = kVec5, .name = "foo_by_foo_systame"},
                                             {.in = kParam6, .out = kVec6, .name = "float_system"},
                                             {.in = kParam7, .out = kVec7, .name = "null_execute"}}};

// не изменяется (определяет какие тесты будем запускать - сек и мпай )
const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SavvaDConjugentGradientsSEQ, InType>(kTestParam, PPC_SETTINGS_savva_d_conjugent_gradients),
    ppc::util::AddFuncTask<SavvaDConjugentGradientsMPI, InType>(kTestParam, PPC_SETTINGS_savva_d_conjugent_gradients));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SavvaDConjugentGradientsFuncTests::PrintFuncTestName<SavvaDConjugentGradientsFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, SavvaDConjugentGradientsFuncTests, kGtestValues,
                         kPerfTestName);  // здесь запуск тестов

}  // namespace

}  // namespace savva_d_conjugent_gradients
