#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>

#include "denisov_a_min_val_row_matrix/common/include/common.hpp"
#include "denisov_a_min_val_row_matrix/mpi/include/ops_mpi.hpp"
#include "denisov_a_min_val_row_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace denisov_a_min_val_row_matrix {

class DenisovAMinValRowMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "x" + std::to_string(std::get<1>(test_param)) + "_" +
           std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType test_params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int rows = std::get<0>(test_params);
    const int cols = std::get<1>(test_params);

    unsigned int seed = 4;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distrib(-1000, 1000);
    input_data_.resize(rows);
    for (int i = 0; i < rows; ++i) {
      input_data_[i].resize(cols);
      std::generate(input_data_[i].begin(), input_data_[i].end(), [&gen, &distrib]() { return distrib(gen); });
    }

    expected_output_data_.resize(rows);
    for (int i = 0; i < rows; ++i) {
      expected_output_data_[i] = *std::min_element(input_data_[i].begin(), input_data_[i].end());
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    return (expected_output_data_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_data_;
};

namespace {

TEST_P(DenisovAMinValRowMatrixFuncTests, RunFuncTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParams = {
    std::make_tuple(1, 1, "1x1"),
    std::make_tuple(5, 5, "5x5"),
    std::make_tuple(20, 100, "20x100"),
    std::make_tuple(120, 100, "120x100"),
    std::make_tuple(1000, 1000, "1000x1000"),
};

const auto kTestTaskList = std::tuple_cat(
    ppc::util::AddFuncTask<DenisovAMinValRowMatrixSEQ, InType>(kTestParams, PPC_SETTINGS_denisov_a_min_val_row_matrix),
    ppc::util::AddFuncTask<DenisovAMinValRowMatrixMPI, InType>(kTestParams, PPC_SETTINGS_denisov_a_min_val_row_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTaskList);

const auto kTestFuncTestName = DenisovAMinValRowMatrixFuncTests::PrintFuncTestName<DenisovAMinValRowMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(MinValRowMatrixTests, DenisovAMinValRowMatrixFuncTests, kGtestValues, kTestFuncTestName);

}  // namespace

}  // namespace denisov_a_min_val_row_matrix
