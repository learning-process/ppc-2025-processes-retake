#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <string>
#include <tuple>
#include <vector>

#include "nozdrin_a_mult_matr_cannon/common/include/common.hpp"
#include "nozdrin_a_mult_matr_cannon/mpi/include/ops_mpi.hpp"
#include "nozdrin_a_mult_matr_cannon/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nozdrin_a_mult_matr_cannon {

class NozdrinAMultMatrCannonFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param) + "_" + std::to_string(std::get<0>(test_param)) + "x" +
           std::to_string(std::get<0>(test_param)) + "_Elems_Up_To_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    size_t rc = std::get<0>(params);
    size_t size = rc * rc;

    int up_to = std::get<1>(params);

    std::vector<double> a(rc * rc);
    std::vector<double> b(rc * rc);

    for (size_t i = 0; i < size; i++) {
      a[i] = static_cast<double>(up_to - static_cast<int>(i));
      b[i] = static_cast<double>(i) * static_cast<double>(up_to) / static_cast<double>(size - 1);
    }

    input_data_ = std::make_tuple(rc, a, b);

    std::vector<double> c(size, 0.0);
    LocalMatrixMultiply(a, b, c, static_cast<int>(rc));
    expected_output_ = c;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (expected_output_.size() != output_data.size()) {
      return false;
    }
    const double epsilon = 1e-7;
    for (size_t i = 0; i < expected_output_.size(); ++i) {
      if (std::abs(expected_output_[i] - output_data[i]) > epsilon) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  static void LocalMatrixMultiply(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c,
                                  int n) {
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < n; ++k) {
        double temp = a[(i * n) + k];
        for (int j = 0; j < n; ++j) {
          c[(i * n) + j] += temp * b[(k * n) + j];
        }
      }
    }
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

class NozdrinAMultMatrCannonSeqTests : public NozdrinAMultMatrCannonFuncTests {};

class NozdrinAMultMatrCannonMpiTests : public NozdrinAMultMatrCannonFuncTests {};

namespace {

TEST_P(NozdrinAMultMatrCannonSeqTests, SeqTest) {
  ExecuteTest(GetParam());
}

TEST_P(NozdrinAMultMatrCannonMpiTests, MpiTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kSeqParams = {
    std::make_tuple(1, 100, "Small_matrix"),        std::make_tuple(2, 100, "Small_2_matrix"),
    std::make_tuple(3, 150, "Medium_small_matrix"), std::make_tuple(4, 200, "Medium_matrix"),
    std::make_tuple(6, 500, "Medium_large_matrix"), std::make_tuple(10, 1000, "Large_matrix"),
    std::make_tuple(12, 2000, "Extra_large_matrix")};

const std::array<TestType, 7> kMpiParams = {
    std::make_tuple(1, 100, "Small_matrix"),        std::make_tuple(2, 100, "Small_2_matrix"),
    std::make_tuple(3, 150, "Medium_small_matrix"), std::make_tuple(4, 200, "Medium_matrix"),
    std::make_tuple(6, 500, "Medium_large_matrix"), std::make_tuple(8, 1000, "Large_matrix"),
    std::make_tuple(12, 2000, "Extra_large_matrix")};

const auto kSeqTasks =
    ppc::util::AddFuncTask<NozdrinAMultMatrCannonSEQ, InType>(kSeqParams, PPC_SETTINGS_nozdrin_a_mult_matr_cannon);
const auto kMpiTasks =
    ppc::util::AddFuncTask<NozdrinAMultMatrCannonMPI, InType>(kMpiParams, PPC_SETTINGS_nozdrin_a_mult_matr_cannon);

const auto kFuncTestName = NozdrinAMultMatrCannonFuncTests::PrintFuncTestName<NozdrinAMultMatrCannonFuncTests>;

INSTANTIATE_TEST_SUITE_P(SeqTests, NozdrinAMultMatrCannonSeqTests, ppc::util::ExpandToValues(kSeqTasks), kFuncTestName);

INSTANTIATE_TEST_SUITE_P(MpiTests, NozdrinAMultMatrCannonMpiTests, ppc::util::ExpandToValues(kMpiTasks), kFuncTestName);

}  // namespace

}  // namespace nozdrin_a_mult_matr_cannon
