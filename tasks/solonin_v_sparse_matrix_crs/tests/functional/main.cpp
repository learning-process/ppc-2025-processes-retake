#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <string>
#include <tuple>
#include <vector>

#include "solonin_v_sparse_matrix_crs/common/include/common.hpp"
#include "solonin_v_sparse_matrix_crs/mpi/include/ops_mpi.hpp"
#include "solonin_v_sparse_matrix_crs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace solonin_v_sparse_matrix_crs {

namespace {

void DenseToCRS(const std::vector<std::vector<double>> &dense, std::vector<double> &vals,
                std::vector<int> &cols, std::vector<int> &ptr) {
  vals.clear();
  cols.clear();
  ptr.clear();
  ptr.push_back(0);
  for (const auto &row : dense) {
    for (size_t j = 0; j < row.size(); j++) {
      if (std::abs(row[j]) > 1e-12) {
        cols.push_back(static_cast<int>(j));
        vals.push_back(row[j]);
      }
    }
    ptr.push_back(static_cast<int>(vals.size()));
  }
}

bool CRSToDense(const std::vector<double> &vals, const std::vector<int> &cols,
                const std::vector<int> &ptr, int rows, int ncols,
                std::vector<std::vector<double>> &dense) {
  if (ptr.empty() || static_cast<int>(ptr.size()) != rows + 1) {
    dense.assign(rows, std::vector<double>(ncols, 0.0));
    return false;
  }
  dense.assign(rows, std::vector<double>(ncols, 0.0));
  for (int i = 0; i < rows; i++) {
    for (int k = ptr[i]; k < ptr[i + 1]; k++) {
      int j = cols[k];
      if (j >= 0 && j < ncols) dense[i][j] = vals[k];
    }
  }
  return true;
}

}  // namespace

class SoloninVCRSFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &tp) {
    return std::to_string(std::get<0>(tp)) + "_crs_mul";
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    dense_a_ = std::get<1>(params);
    dense_b_ = std::get<2>(params);
    dense_exp_ = std::get<3>(params);
    if (dense_a_.empty()) {
      ptr_a_ = {0};
    } else {
      DenseToCRS(dense_a_, vals_a_, cols_a_, ptr_a_);
    }
    if (dense_b_.empty()) {
      ptr_b_ = {0};
    } else {
      DenseToCRS(dense_b_, vals_b_, cols_b_, ptr_b_);
    }
  }

  bool CheckTestOutputData(OutType &out) final {
    auto &[vals, cols, ptr] = out;
    int rows = static_cast<int>(dense_exp_.size());
    int ncols = dense_exp_.empty() ? 0 : static_cast<int>(dense_exp_[0].size());

    if (ptr.empty()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) return true;
      for (const auto &r : dense_exp_)
        for (double v : r)
          if (std::abs(v) > 1e-12) return false;
      return true;
    }

    std::vector<std::vector<double>> res;
    if (!CRSToDense(vals, cols, ptr, rows, ncols, res)) return false;
    const double tol = 1e-10;
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < ncols; j++)
        if (std::abs(res[i][j] - dense_exp_[i][j]) > tol) return false;
    return true;
  }

  InType GetTestInputData() final {
    int rows = static_cast<int>(dense_a_.size());
    int ca = dense_a_.empty() ? 0 : static_cast<int>(dense_a_[0].size());
    int cb = dense_b_.empty() ? 0 : static_cast<int>(dense_b_[0].size());
    return std::make_tuple(vals_a_, cols_a_, ptr_a_, vals_b_, cols_b_, ptr_b_, rows, ca, cb);
  }

 private:
  std::vector<std::vector<double>> dense_a_, dense_b_, dense_exp_;
  std::vector<double> vals_a_, vals_b_;
  std::vector<int> cols_a_, cols_b_;
  std::vector<int> ptr_a_, ptr_b_;
};

namespace {

TEST_P(SoloninVCRSFuncTests, FunctionalTests) { ExecuteTest(GetParam()); }

const std::array<TestType, 20> kTests = {
    std::make_tuple(1, std::vector<std::vector<double>>{{1, 2}, {3, 4}},
                    std::vector<std::vector<double>>{{5, 6}, {7, 8}},
                    std::vector<std::vector<double>>{{19, 22}, {43, 50}}),
    std::make_tuple(2, std::vector<std::vector<double>>{{1, 0}, {0, 1}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}}),
    std::make_tuple(3, std::vector<std::vector<double>>{{1, 1}, {1, 1}},
                    std::vector<std::vector<double>>{{1, 1}, {1, 1}},
                    std::vector<std::vector<double>>{{2, 2}, {2, 2}}),
    std::make_tuple(4, std::vector<std::vector<double>>{{2, 0}, {0, 2}},
                    std::vector<std::vector<double>>{{3, 0}, {0, 3}},
                    std::vector<std::vector<double>>{{6, 0}, {0, 6}}),
    std::make_tuple(5, std::vector<std::vector<double>>{{1, 2, 3}},
                    std::vector<std::vector<double>>{{4}, {5}, {6}},
                    std::vector<std::vector<double>>{{32}}),
    std::make_tuple(6, std::vector<std::vector<double>>{{1}, {2}, {3}},
                    std::vector<std::vector<double>>{{4, 5, 6}},
                    std::vector<std::vector<double>>{{4, 5, 6}, {8, 10, 12}, {12, 15, 18}}),
    std::make_tuple(7, std::vector<std::vector<double>>{{1}},
                    std::vector<std::vector<double>>{{1}},
                    std::vector<std::vector<double>>{{1}}),
    std::make_tuple(8, std::vector<std::vector<double>>{{0, 0}, {0, 0}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}},
                    std::vector<std::vector<double>>{{0, 0}, {0, 0}}),
    std::make_tuple(9, std::vector<std::vector<double>>{{2}},
                    std::vector<std::vector<double>>{{3}},
                    std::vector<std::vector<double>>{{6}}),
    std::make_tuple(10, std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}},
                    std::vector<std::vector<double>>{{7, 8}, {9, 10}, {11, 12}},
                    std::vector<std::vector<double>>{{58, 64}, {139, 154}}),
    std::make_tuple(11, std::vector<std::vector<double>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                    std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                    std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}),
    std::make_tuple(12, std::vector<std::vector<double>>{{-1, -2}, {-3, -4}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}},
                    std::vector<std::vector<double>>{{-7, -10}, {-15, -22}}),
    std::make_tuple(13, std::vector<std::vector<double>>{{1, -2}, {3, -4}},
                    std::vector<std::vector<double>>{{-1, 2}, {-3, 4}},
                    std::vector<std::vector<double>>{{5, -6}, {9, -10}}),
    std::make_tuple(14, std::vector<std::vector<double>>{{0.5, 0.5}, {0.5, 0.5}},
                    std::vector<std::vector<double>>{{2, 4}, {6, 8}},
                    std::vector<std::vector<double>>{{4, 6}, {4, 6}}),
    std::make_tuple(15, std::vector<std::vector<double>>{{1, 0, 2, 0}, {0, 3, 0, 4}, {5, 0, 0, 6}, {0, 7, 8, 0}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}, {7, 8}},
                    std::vector<std::vector<double>>{{11, 14}, {37, 44}, {47, 58}, {61, 76}}),
    std::make_tuple(16, std::vector<std::vector<double>>{{1, 0, 0, 2}, {0, 3, 0, 0}, {0, 0, 4, 0}},
                    std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
                    std::vector<std::vector<double>>{{21, 24, 27}, {12, 15, 18}, {28, 32, 36}}),
    std::make_tuple(17, std::vector<std::vector<double>>{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}},
                    std::vector<std::vector<double>>{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}},
                    std::vector<std::vector<double>>{{1, 1, 1}, {4, 4, 4}, {9, 9, 9}}),
    std::make_tuple(18, std::vector<std::vector<double>>{{0, 0, 0}, {1, 2, 3}, {0, 0, 0}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}},
                    std::vector<std::vector<double>>{{0, 0}, {22, 28}, {0, 0}}),
    std::make_tuple(19, std::vector<std::vector<double>>{{0.1, 0.2}, {0.3, 0.4}},
                    std::vector<std::vector<double>>{{5, 6}, {7, 8}},
                    std::vector<std::vector<double>>{{1.9, 2.2}, {4.3, 5.0}}),
    std::make_tuple(20, std::vector<std::vector<double>>{{1, 2}, {3, 4}},
                    std::vector<std::vector<double>>{{0, 0}, {0, 0}},
                    std::vector<std::vector<double>>{{0, 0}, {0, 0}}),
};

const auto kTaskList = std::tuple_cat(
    ppc::util::AddFuncTask<SoloninVSparseMulCRSMPI, InType>(kTests, PPC_SETTINGS_solonin_v_sparse_matrix_crs),
    ppc::util::AddFuncTask<SoloninVSparseMulCRSSEQ, InType>(kTests, PPC_SETTINGS_solonin_v_sparse_matrix_crs));

const auto kGtestValues = ppc::util::ExpandToValues(kTaskList);
const auto kTestName = SoloninVCRSFuncTests::PrintFuncTestName<SoloninVCRSFuncTests>;

INSTANTIATE_TEST_SUITE_P(CRSMultiply, SoloninVCRSFuncTests, kGtestValues, kTestName);

}  // namespace

}  // namespace solonin_v_sparse_matrix_crs
