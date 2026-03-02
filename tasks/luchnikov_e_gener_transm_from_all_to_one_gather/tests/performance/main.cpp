#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/seq/include/ops_seq.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

namespace {
size_t GetTypeSizeSeq(MPI_Datatype datatype) {
  if (datatype == MPI_INT) {
    return sizeof(int);
  }
  if (datatype == MPI_FLOAT) {
    return sizeof(float);
  }
  if (datatype == MPI_DOUBLE) {
    return sizeof(double);
  }
  return 0;
}

std::string GetTaskName(const std::tuple<std::string, std::string, int> &param) {
  return std::get<1>(param);
}
}  // namespace

class LuchnikovETransmFrAllToOneGatherPerfTests
    : public ::testing::TestWithParam<std::tuple<std::string, std::string, int>> {
 protected:
  static const size_t kDataCount = 100000;
  MPI_Datatype data_type = MPI_INT;

  void SetUp() override {
    const size_t type_size = sizeof(int);
    std::vector<char> data(kDataCount * type_size);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto *data_ptr = reinterpret_cast<int *>(data.data());
    for (size_t i = 0; i < kDataCount; ++i) {
      data_ptr[i] = static_cast<int>((static_cast<size_t>(rank) * kDataCount) + i);
    }

    const int root = 0;
    input_data_ = GatherInput{.data = data, .count = static_cast<int>(kDataCount), .datatype = data_type, .root = root};
  }

  bool CheckOutputData(OutType &output_data) {
    const auto &input = input_data_;

    const auto &params = GetParam();
    const std::string task_name = std::get<1>(params);
    const bool is_mpi = task_name.find("MPI") != std::string::npos;

    int world_size = 1;
    int rank = 0;
    if (is_mpi) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != input.root) {
        return true;
      }
    }

    const size_t type_size = GetTypeSizeSeq(input.datatype);
    const size_t expected_size = static_cast<size_t>(input.count) * static_cast<size_t>(world_size) * type_size;

    if (output_data.size() != expected_size) {
      return false;
    }

    if (rank == input.root && input.datatype == MPI_INT) {
      const int *out_ptr = reinterpret_cast<const int *>(output_data.data());
      for (int i = 0; i < world_size; ++i) {
        for (size_t j = 0; j < kDataCount; ++j) {
          int expected = static_cast<int>(i * static_cast<int>(kDataCount) + static_cast<int>(j));
          int actual = out_ptr[i * static_cast<int>(kDataCount) + static_cast<int>(j)];
          if (actual != expected) {
            return false;
          }
        }
      }
    }

    return true;
  }

  GatherInput GetInputData() {
    return input_data_;
  }

  void ExecuteTest() {
    const auto &params = GetParam();
    const std::string task_type = std::get<1>(params);

    if (task_type.find("MPI") != std::string::npos) {
      LuchnikovETransmFrAllToOneGatherMPI task(input_data_);
      ASSERT_TRUE(task.Validation());
      ASSERT_TRUE(task.PreProcessing());
      ASSERT_TRUE(task.Run());
      ASSERT_TRUE(task.PostProcessing());

      auto output = task.GetOutput();
      EXPECT_TRUE(CheckOutputData(output));
    } else {
      LuchnikovETransmFrAllToOneGatherSEQ task(input_data_);
      ASSERT_TRUE(task.Validation());
      ASSERT_TRUE(task.PreProcessing());
      ASSERT_TRUE(task.Run());
      ASSERT_TRUE(task.PostProcessing());

      auto output = task.GetOutput();
      EXPECT_TRUE(CheckOutputData(output));
    }
  }

 private:
  GatherInput input_data_;
};

TEST_P(LuchnikovETransmFrAllToOneGatherPerfTests, RunPerfModes) {
  ExecuteTest();
}

std::vector<std::tuple<std::string, std::string, int>> CreatePerfTestParams() {
  std::vector<std::tuple<std::string, std::string, int>> params;

  params.push_back(std::make_tuple("luchnikov_e_gener_transm_from_all_to_one_gather", "MPI", 0));

  params.push_back(std::make_tuple("luchnikov_e_gener_transm_from_all_to_one_gather", "SEQ", 0));

  return params;
}

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuchnikovETransmFrAllToOneGatherPerfTests,
                         ::testing::ValuesIn(CreatePerfTestParams()),
                         [](const testing::TestParamInfo<LuchnikovETransmFrAllToOneGatherPerfTests::ParamType> &info) {
                           return std::get<1>(info.param);
                         });

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
