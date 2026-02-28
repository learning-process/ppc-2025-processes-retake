// [file name]: tests/performance/main.cpp
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
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnilkovEMaxValInColOfMatPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr size_t kCount = 1000;
  InType input_data_{};

  void SetUp() override {
    input_data_.resize(kCount);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 1000);

    for (size_t i = 0; i < kCount; ++i) {
      input_data_[i].resize(kCount);
      for (size_t j = 0; j < kCount; ++j) {
        input_data_[i][j] = dist(gen);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Получаем ранг через MPI напрямую
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Для MPI задач проверяем только на нулевом процессе
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
};

TEST_P(LuchnilkovEMaxValInColOfMatPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LuchnilkovEMaxValInColOfMatMPI, LuchnilkovEMaxValInColOfMatSEQ>(
        PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuchnilkovEMaxValInColOfMatPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuchnilkovEMaxValInColOfMatPerfTestProcesses, kGtestValues,
                         kPerfTestName);  // NOLINT

}  // namespace

}  // namespace luchnikov_e_max_val_in_col_of_mat
