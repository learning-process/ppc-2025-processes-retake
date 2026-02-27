#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "Nazarova_K_Gaussian_vert_scheme/common/include/common.hpp"
#include "Nazarova_K_Gaussian_vert_scheme/mpi/include/ops_mpi.hpp"
#include "Nazarova_K_Gaussian_vert_scheme/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nazarova_k_gaussian_vert_scheme_processes {
namespace {

constexpr double kTol = 1e-7;

Input MakeSystem(int n, unsigned seed, std::vector<double> *x_expected) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  std::vector<double> x(static_cast<std::size_t>(n));
  for (double &v : x) {
    v = dist(gen);
  }
  if (x_expected != nullptr) {
    *x_expected = x;
  }

  // Random diagonally-dominant matrix to guarantee non-singularity
  std::vector<double> a(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  for (int i = 0; i < n; i++) {
    double row_sum = 0.0;
    for (int j = 0; j < n; j++) {
      double v = dist(gen);
      a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(j)] = v;
      row_sum += std::abs(v);
    }
    // make diagonal strictly dominant
    a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(i)] += row_sum + 1.0;
  }

  // b = A * x
  std::vector<double> b(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(j)] *
             x[static_cast<std::size_t>(j)];
    }
    b[static_cast<std::size_t>(i)] = sum;
  }

  Input in;
  in.n = n;
  in.augmented.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n + 1), 0.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      in.augmented[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(j)] =
          a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(j)];
    }
    in.augmented[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(n)] =
        b[static_cast<std::size_t>(i)];
  }
  return in;
}

bool VectorsNear(const std::vector<double> &a, const std::vector<double> &b, double tol) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > tol) {
      return false;
    }
  }
  return true;
}

}  // namespace

class NazarovaKGaussianVertSchemeRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int n = std::get<0>(params);
    input_data_ = MakeSystem(n, 123U + static_cast<unsigned>(n), &expected_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return VectorsNear(output_data, expected_, kTol);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_;
};

namespace {

TEST_P(NazarovaKGaussianVertSchemeRunFuncTests, SolveSLAE) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "n3"), std::make_tuple(5, "n5"),
                                            std::make_tuple(10, "n10")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<NazarovaKGaussianVertSchemeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_Nazarova_K_Gaussian_vert_scheme),
                                           ppc::util::AddFuncTask<NazarovaKGaussianVertSchemeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_Nazarova_K_Gaussian_vert_scheme));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    NazarovaKGaussianVertSchemeRunFuncTests::PrintFuncTestName<NazarovaKGaussianVertSchemeRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(GaussVertStripTests, NazarovaKGaussianVertSchemeRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace nazarova_k_gaussian_vert_scheme_processes
