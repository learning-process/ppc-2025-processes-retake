#include <gtest/gtest.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <cstddef>

#include "savva_d_min_elem_vec//common/include/common.hpp"
#include "savva_d_min_elem_vec/mpi/include/ops_mpi.hpp"
#include "savva_d_min_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace savva_d_min_elem_vec {

class SavvaDMinElemVecPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  int expected_min_ = 0;

  void SetUp() override {
    // Чтение данных из файла

    std::string file_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_savva_d_min_elem_vec, "data.txt");
    std::ifstream file(file_path);

    if (!file.is_open()) {
      throw std::runtime_error("Cannot open test data file");
    }

    // Читаем размер вектора
    int vector_size = 0;
    file >> vector_size;

    file >> expected_min_;

    // Читаем данные вектора
    input_data_.resize(static_cast<size_t>(vector_size) * 51);
    for (int i = 0; i < vector_size; ++i) {
      file >> input_data_[i];
    }

    for (int i = 1; i < 51; ++i) {
      for (int j = 0; j < vector_size; ++j) {
        input_data_[(vector_size * i) + j] = input_data_[(vector_size * (i - 1)) + j] - 50000;
      }
    }

    file.close();

    // Проверяем что данные загружены корректно
    if (input_data_.empty()) {
      throw std::runtime_error("Test data is empty!");
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_min_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SavvaDMinElemVecPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SavvaDMinElemVecMPI, SavvaDMinElemVecSEQ>(PPC_SETTINGS_savva_d_min_elem_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SavvaDMinElemVecPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SavvaDMinElemVecPerfTest, kGtestValues, kPerfTestName);

}  // namespace savva_d_min_elem_vec
