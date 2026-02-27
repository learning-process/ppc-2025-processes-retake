#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zyuzin_n_multiplication_matrix_horiz/common/include/common.hpp"
#include "zyuzin_n_multiplication_matrix_horiz/mpi/include/ops_mpi.hpp"
#include "zyuzin_n_multiplication_matrix_horiz/seq/include/ops_seq.hpp"

namespace zyuzin_n_multiplication_matrix_horiz {

class ZyuzinNMultiplicationMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_test";
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    matrix_a_ = std::get<1>(params);
    matrix_b_ = std::get<2>(params);
    expected_ = std::get<3>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_.size() ||
        (!output_data.empty() && output_data[0].size() != expected_[0].size())) {
      return false;
    }

    const double k_eps = 1e-10;
    for (size_t i = 0; i < expected_.size(); i++) {
      for (size_t j = 0; j < expected_[0].size(); j++) {
        if (std::fabs(output_data[i][j] - expected_[i][j]) > k_eps) {
          return false;
        }
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return std::make_pair(matrix_a_, matrix_b_);
  }

 private:
  std::vector<std::vector<double>> matrix_a_;
  std::vector<std::vector<double>> matrix_b_;
  OutType expected_;
};

namespace {

TEST_P(ZyuzinNMultiplicationMatrixFuncTests, FunctionalTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 22> kTestParam = {
    std::make_tuple(1, std::vector<std::vector<double>>{{2, 0}, {1, 3}},
                    std::vector<std::vector<double>>{{4, 1}, {2, 2}},
                    std::vector<std::vector<double>>{{8, 2}, {10, 7}}),

    std::make_tuple(2, std::vector<std::vector<double>>{{1, 0}, {0, 1}},
                    std::vector<std::vector<double>>{{5, 9}, {4, 3}}, std::vector<std::vector<double>>{{5, 9}, {4, 3}}),

    std::make_tuple(3, std::vector<std::vector<double>>{{2, 2}, {2, 2}},
                    std::vector<std::vector<double>>{{3, 3}, {3, 3}},
                    std::vector<std::vector<double>>{{12, 12}, {12, 12}}),

    std::make_tuple(4, std::vector<std::vector<double>>{{3, 0}, {0, 4}},
                    std::vector<std::vector<double>>{{5, 0}, {0, 2}},
                    std::vector<std::vector<double>>{{15, 0}, {0, 8}}),

    std::make_tuple(5, std::vector<std::vector<double>>{{1, 4, 2}}, std::vector<std::vector<double>>{{2}, {3}, {5}},
                    std::vector<std::vector<double>>{{24}}),

    std::make_tuple(6, std::vector<std::vector<double>>{{2}, {1}, {0}}, std::vector<std::vector<double>>{{3, 1, 2}},
                    std::vector<std::vector<double>>{{6, 2, 4}, {3, 1, 2}, {0, 0, 0}}),

    std::make_tuple(7, std::vector<std::vector<double>>{{1, 2, 0}, {0, 1, 2}},
                    std::vector<std::vector<double>>{{1, 0}, {2, 1}, {0, 2}},
                    std::vector<std::vector<double>>{{5, 2}, {2, 5}}),

    std::make_tuple(8, std::vector<std::vector<double>>{{5}}, std::vector<std::vector<double>>{{6}},
                    std::vector<std::vector<double>>{{30}}),

    std::make_tuple(9, std::vector<std::vector<double>>{{0, 0}, {0, 0}},
                    std::vector<std::vector<double>>{{7, 8}, {9, 10}},
                    std::vector<std::vector<double>>{{0, 0}, {0, 0}}),

    std::make_tuple(10, std::vector<std::vector<double>>{{-2}}, std::vector<std::vector<double>>{{4}},
                    std::vector<std::vector<double>>{{-8}}),

    std::make_tuple(11, std::vector<std::vector<double>>{{1, 1, 1}, {2, 2, 2}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}},
                    std::vector<std::vector<double>>{{9, 12}, {18, 24}}),

    std::make_tuple(12, std::vector<std::vector<double>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                    std::vector<std::vector<double>>{{10, 20, 30}, {40, 50, 60}, {70, 80, 90}},
                    std::vector<std::vector<double>>{{10, 20, 30}, {40, 50, 60}, {70, 80, 90}}),

    std::make_tuple(13, std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}},
                    std::vector<std::vector<double>>{{1, 0, 1}, {0, 1, 0}},
                    std::vector<std::vector<double>>{{1, 2, 1}, {3, 4, 3}, {5, 6, 5}}),

    std::make_tuple(14, std::vector<std::vector<double>>{{7, 3}, {2, 1}},
                    std::vector<std::vector<double>>{{1, 0}, {0, 1}}, std::vector<std::vector<double>>{{7, 3}, {2, 1}}),

    std::make_tuple(15, std::vector<std::vector<double>>{{0, 1}, {1, 0}},
                    std::vector<std::vector<double>>{{2, 3}, {4, 5}}, std::vector<std::vector<double>>{{4, 5}, {2, 3}}),

    std::make_tuple(16, std::vector<std::vector<double>>{{5, 5}, {5, 5}},
                    std::vector<std::vector<double>>{{2, 2}, {2, 2}},
                    std::vector<std::vector<double>>{{20, 20}, {20, 20}}),

    std::make_tuple(17, std::vector<std::vector<double>>{{0}}, std::vector<std::vector<double>>{{100}},
                    std::vector<std::vector<double>>{{0}}),

    std::make_tuple(18, std::vector<std::vector<double>>{{1, 0}, {0, 0}},
                    std::vector<std::vector<double>>{{0, 0}, {0, 1}}, std::vector<std::vector<double>>{{0, 0}, {0, 0}}),

    std::make_tuple(19, std::vector<std::vector<double>>{{0.5, 0}, {0, 0.5}},
                    std::vector<std::vector<double>>{{4, 8}, {2, 6}}, std::vector<std::vector<double>>{{2, 4}, {1, 3}}),

    std::make_tuple(20, std::vector<std::vector<double>>{{10, 20}, {30, 40}},
                    std::vector<std::vector<double>>{{0, 0}, {0, 0}}, std::vector<std::vector<double>>{{0, 0}, {0, 0}}),

    std::make_tuple(21, std::vector<std::vector<double>>{{0.5, 0.5}, {1.0, 1.0}},
                    std::vector<std::vector<double>>{{2.0, 2.0}, {2.0, 2.0}},
                    std::vector<std::vector<double>>{{2.0, 2.0}, {4.0, 4.0}}),

    std::make_tuple(22, std::vector<std::vector<double>>{{0.3, 0.7}}, std::vector<std::vector<double>>{{0.5}, {0.5}},
                    std::vector<std::vector<double>>{{0.5}})};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ZyuzinNMultiplicationMatrixMPI, InType>(
                                               kTestParam, PPC_SETTINGS_zyuzin_n_multiplication_matrix_horiz),
                                           ppc::util::AddFuncTask<ZyuzinNMultiplicationMatrixSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_zyuzin_n_multiplication_matrix_horiz));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    ZyuzinNMultiplicationMatrixFuncTests::PrintFuncTestName<ZyuzinNMultiplicationMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(MultiplicationMatrixTests, ZyuzinNMultiplicationMatrixFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zyuzin_n_multiplication_matrix_horiz
