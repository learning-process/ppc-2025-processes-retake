// [file name]: tests/functional/main.cpp
#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnilkovEMaxValInColOfMatFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    size_t size = std::get<0>(params);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);

    input_data_.resize(size);
    for (size_t i = 0; i < size; ++i) {
      input_data_[i].resize(size);
      for (size_t j = 0; j < size; ++j) {
        input_data_[i][j] = dist(gen);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (this->GetParamType() == ppc::util::TestType::MPI && rank != 0) {
      return true;
    }

    if (input_data_.empty() || input_data_[0].empty()) {
      return output_data.empty();
    }

    size_t cols = input_data_[0].size();
    OutType expected(cols, std::numeric_limits<int>::min());

    for (const auto &row : input_data_) {
      for (size_t j = 0; j < cols; ++j) {
        expected[j] = std::max(expected[j], row[j]);
      }
    }

    return expected == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(LuchnilkovEMaxValInColOfMatFuncTestsProcesses, MaxInColTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(1, "1"),   std::make_tuple(2, "2"),   std::make_tuple(3, "3"),   std::make_tuple(5, "5"),
    std::make_tuple(7, "7"),   std::make_tuple(10, "10"), std::make_tuple(15, "15"), std::make_tuple(20, "20"),
    std::make_tuple(25, "25"), std::make_tuple(30, "30")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LuchnilkovEMaxValInColOfMatMPI, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat),
                                           ppc::util::AddFuncTask<LuchnilkovEMaxValInColOfMatSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    LuchnilkovEMaxValInColOfMatFuncTestsProcesses::PrintFuncTestName<LuchnilkovEMaxValInColOfMatFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MaxInColTests, LuchnilkovEMaxValInColOfMatFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace luchnikov_e_max_val_in_col_of_mat
