#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "cheremkhin_a_gaus_vert/common/include/common.hpp"
#include "cheremkhin_a_gaus_vert/mpi/include/ops_mpi.hpp"
#include "cheremkhin_a_gaus_vert/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace cheremkhin_a_gaus_vert {

[[maybe_unused]] static void PrintTo(const Input &in, std::ostream *os) {
  *os << "{n=" << in.n << ", a_size=" << in.a.size() << ", b_size=" << in.b.size() << "}";
}

static void PrintTo(const ppc::util::FuncTestParam<InType, OutType, TestType> &param, std::ostream *os) {
  const auto &test_name = std::get<1>(param);
  const auto &test_case = std::get<2>(param);
  *os << "{name=" << test_name << ", n=" << std::get<0>(test_case).n << "}";
}

namespace {

Input MakeInputFromAxb(const std::vector<double> &a, const std::vector<double> &b, int n) {
  Input in;
  in.n = n;
  in.a = a;
  in.b = b;
  return in;
}

std::vector<double> MulAx(const std::vector<double> &a, const std::vector<double> &x, int n) {
  std::vector<double> b(static_cast<std::size_t>(n), 0.0);
  for (int row = 0; row < n; ++row) {
    double s = 0.0;
    for (int col = 0; col < n; ++col) {
      s += a[(static_cast<std::size_t>(row) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(col)] *
           x[static_cast<std::size_t>(col)];
    }
    b[static_cast<std::size_t>(row)] = s;
  }
  return b;
}

}  // namespace

class CheremkhinAGausVertRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param).n);
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<2>(GetParam());
    input_data_ = std::get<0>(params);
    correct_answer_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != correct_answer_.size()) {
      return false;
    }
    const double eps = 1e-8;
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - correct_answer_[i]) > eps) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  OutType correct_answer_;
  InType input_data_;
};

namespace {

TEST_P(CheremkhinAGausVertRunFuncTestsProcesses, SolveLinearSystem) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(MakeInputFromAxb(std::vector<double>{2.0, 1.0, 5.0, 7.0}, std::vector<double>{3.0, 12.0}, 2),
                    OutType{1.0, 1.0}),
    std::make_tuple(MakeInputFromAxb(std::vector<double>{3.0, 2.0, -1.0, 2.0, -2.0, 4.0, -1.0, 0.5, -1.0},
                                     MulAx(std::vector<double>{3.0, 2.0, -1.0, 2.0, -2.0, 4.0, -1.0, 0.5, -1.0},
                                           std::vector<double>{1.0, -2.0, -2.0}, 3),
                                     3),
                    OutType{1.0, -2.0, -2.0}),
    std::make_tuple(MakeInputFromAxb(std::vector<double>{5.0}, std::vector<double>{10.0}, 1), OutType{2.0})};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<CheremkhinAGausVertMPI, InType>(kTestParam, PPC_SETTINGS_cheremkhin_a_gaus_vert),
    ppc::util::AddFuncTask<CheremkhinAGausVertSEQ, InType>(kTestParam, PPC_SETTINGS_cheremkhin_a_gaus_vert));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    CheremkhinAGausVertRunFuncTestsProcesses::PrintFuncTestName<CheremkhinAGausVertRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GausVertTest, CheremkhinAGausVertRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace cheremkhin_a_gaus_vert
