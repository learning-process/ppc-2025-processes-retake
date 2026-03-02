#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "salena_s_matrix_vector_mult/common/include/common.hpp"
#include "salena_s_matrix_vector_mult/mpi/include/ops_mpi.hpp"
#include "salena_s_matrix_vector_mult/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace salena_s_matrix_vector_mult {

class MatVecMultFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int rows = std::get<0>(params);
    int cols = std::get<1>(params);
    input_data_.rows = rows;
    input_data_.cols = cols;
    input_data_.matrix.resize(static_cast<std::size_t>(rows * cols));
    input_data_.vec.resize(static_cast<std::size_t>(cols));

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (int i = 0; i < rows * cols; ++i) {
      input_data_.matrix[static_cast<std::size_t>(i)] = dist(gen);
    }
    for (int i = 0; i < cols; ++i) {
      input_data_.vec[static_cast<std::size_t>(i)] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int is_mpi_init = 0;
    MPI_Initialized(&is_mpi_init);
    if (is_mpi_init) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;  // Сравниваем ответы только на рут-процессе!
      }
    }

    if (output_data.size() != static_cast<std::size_t>(input_data_.rows)) {
      return false;
    }

    std::vector<double> expected(input_data_.rows, 0.0);
    for (int i = 0; i < input_data_.rows; ++i) {
      for (int j = 0; j < input_data_.cols; ++j) {
        expected[static_cast<std::size_t>(i)] +=
            input_data_.matrix[static_cast<std::size_t>(i * input_data_.cols + j)] *
            input_data_.vec[static_cast<std::size_t>(j)];
      }
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
      if (std::abs(expected[i] - output_data[i]) > 1e-4) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

TEST_P(MatVecMultFuncTests, RunMult) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(10, 10, "10x10"), std::make_tuple(50, 50, "50x50"),
                                            std::make_tuple(20, 30, "20x30")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<TestTaskMPI, InType>(kTestParam, PPC_SETTINGS_salena_s_matrix_vector_mult),
                   ppc::util::AddFuncTask<TestTaskSEQ, InType>(kTestParam, PPC_SETTINGS_salena_s_matrix_vector_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = MatVecMultFuncTests::PrintFuncTestName<MatVecMultFuncTests>;
INSTANTIATE_TEST_SUITE_P(MatVecMultTests, MatVecMultFuncTests, kGtestValues, kPerfTestName);

}  // namespace salena_s_matrix_vector_mult
