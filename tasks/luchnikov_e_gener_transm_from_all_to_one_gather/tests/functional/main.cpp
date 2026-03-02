#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/seq/include/ops_seq.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

class LuchnikovETransmFrAllToOneGatherFuncTests
    : public ::testing::TestWithParam<std::tuple<std::string, std::string, int, int, MPI_Datatype, std::string>> {
 protected:
  void SetUp() override {
    const auto &count = std::get<2>(GetParam());
    const auto &root = std::get<3>(GetParam());
    const auto &datatype = std::get<4>(GetParam());

    int type_size = 0;
    if (datatype == MPI_INT) {
      type_size = sizeof(int);
    } else if (datatype == MPI_FLOAT) {
      type_size = sizeof(float);
    } else if (datatype == MPI_DOUBLE) {
      type_size = sizeof(double);
    }

    size_t total_size = static_cast<size_t>(count) * static_cast<size_t>(type_size);
    std::vector<char> data(total_size);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (datatype == MPI_INT) {
      auto *data_ptr = reinterpret_cast<int *>(data.data());
      for (int i = 0; i < count; ++i) {
        data_ptr[i] = rank * count + i + 1;
      }
    } else if (datatype == MPI_FLOAT) {
      auto *data_ptr = reinterpret_cast<float *>(data.data());
      for (int i = 0; i < count; ++i) {
        data_ptr[i] = static_cast<float>(rank * count + i) + 0.5F;
      }
    } else if (datatype == MPI_DOUBLE) {
      auto *data_ptr = reinterpret_cast<double *>(data.data());
      for (int i = 0; i < count; ++i) {
        data_ptr[i] = static_cast<double>(rank * count + i) + 0.25;
      }
    }

    input_data_ = GatherInput{.data = data, .count = count, .datatype = datatype, .root = root};
  }

  static int GetTypeSize(MPI_Datatype datatype) {
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

  bool CheckOutputData(OutType &output_data) {
    const auto &input = input_data_;
    const std::string task_type = std::get<1>(GetParam());
    bool is_mpi = (task_type == "MPI");

    int world_size = 1;
    int rank = 0;
    if (is_mpi) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != input.root) {
        return true;
      }
    }

    if (output_data.empty()) {
      return false;
    }

    const auto type_size = static_cast<size_t>(GetTypeSize(input.datatype));
    const size_t expected_size = static_cast<size_t>(input.count) * static_cast<size_t>(world_size) * type_size;

    if (output_data.size() != expected_size) {
      return false;
    }

    // Проверяем корректность собранных данных
    if (rank == input.root) {
      if (input.datatype == MPI_INT) {
        const int *out_ptr = reinterpret_cast<const int *>(output_data.data());
        for (int i = 0; i < world_size; ++i) {
          for (int j = 0; j < input.count; ++j) {
            int expected_value = i * input.count + j + 1;
            int actual_value = out_ptr[i * input.count + j];
            if (actual_value != expected_value) {
              return false;
            }
          }
        }
      }
    }

    return true;
  }

  void ExecuteTest() {
    const std::string task_type = std::get<1>(GetParam());

    if (task_type == "MPI") {
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

namespace {

TEST_P(LuchnikovETransmFrAllToOneGatherFuncTests, GatherCheck) {
  ExecuteTest();
}

// Создаем параметры для тестов
std::vector<std::tuple<std::string, std::string, int, int, MPI_Datatype, std::string>> CreateTestParams() {
  std::vector<std::tuple<std::string, std::string, int, int, MPI_Datatype, std::string>> params;

  std::string task_name = "luchnikov_e_gener_transm_from_all_to_one_gather";

  // Тест 1: count=1, root=0, int
  params.push_back(std::make_tuple(task_name, "MPI", 1, 0, MPI_INT, "SingleElementIntRoot0"));
  params.push_back(std::make_tuple(task_name, "SEQ", 1, 0, MPI_INT, "SingleElementIntRoot0"));

  // Тест 2: count=3, root=0, int
  params.push_back(std::make_tuple(task_name, "MPI", 3, 0, MPI_INT, "SmallIntRoot0"));
  params.push_back(std::make_tuple(task_name, "SEQ", 3, 0, MPI_INT, "SmallIntRoot0"));

  // Тест 3: count=4, root=0, float
  params.push_back(std::make_tuple(task_name, "MPI", 4, 0, MPI_FLOAT, "SmallFloatRoot0"));
  params.push_back(std::make_tuple(task_name, "SEQ", 4, 0, MPI_FLOAT, "SmallFloatRoot0"));

  // Тест 4: count=3, root=0, double
  params.push_back(std::make_tuple(task_name, "MPI", 3, 0, MPI_DOUBLE, "SmallDoubleRoot0"));
  params.push_back(std::make_tuple(task_name, "SEQ", 3, 0, MPI_DOUBLE, "SmallDoubleRoot0"));

  // Тест 5: count=2, root=1, int
  params.push_back(std::make_tuple(task_name, "MPI", 2, 1, MPI_INT, "IntRoot1"));
  params.push_back(std::make_tuple(task_name, "SEQ", 2, 1, MPI_INT, "IntRoot1"));

  // Тест 6: count=2, root=2, int
  params.push_back(std::make_tuple(task_name, "MPI", 2, 2, MPI_INT, "IntRoot2"));
  params.push_back(std::make_tuple(task_name, "SEQ", 2, 2, MPI_INT, "IntRoot2"));

  return params;
}

std::string PrintTestParam(const testing::TestParamInfo<LuchnikovETransmFrAllToOneGatherFuncTests::ParamType> &info) {
  std::string task_type = std::get<1>(info.param);
  std::string test_name = std::get<5>(info.param);
  return task_type + "_" + test_name;
}

INSTANTIATE_TEST_SUITE_P(GatherTests, LuchnikovETransmFrAllToOneGatherFuncTests,
                         ::testing::ValuesIn(CreateTestParams()), PrintTestParam);

TEST(LuchnikovETransmFrAllToOneGatherMPITest, BasicMPIGather) {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int count = 3;
  std::vector<char> data(count * sizeof(int));
  auto *data_ptr = reinterpret_cast<int *>(data.data());
  for (int i = 0; i < count; ++i) {
    data_ptr[i] = rank * count + i + 1;
  }

  GatherInput input{.data = data, .count = count, .datatype = MPI_INT, .root = 0};
  LuchnikovETransmFrAllToOneGatherMPI task(input);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  if (rank == 0) {
    const auto &result = task.GetOutput();
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), static_cast<size_t>(count) * static_cast<size_t>(world_size) * sizeof(int));

    const int *res_ptr = reinterpret_cast<const int *>(result.data());
    for (int i = 0; i < world_size; ++i) {
      for (int j = 0; j < count; ++j) {
        EXPECT_EQ(res_ptr[i * count + j], i * count + j + 1);
      }
    }
  }
}

TEST(LuchnikovETransmFrAllToOneGatherSEQTest, BasicSEQGather) {
  int count = 3;
  std::vector<char> data(count * sizeof(int));
  auto *data_ptr = reinterpret_cast<int *>(data.data());
  for (int i = 0; i < count; ++i) {
    data_ptr[i] = i + 1;
  }

  GatherInput input{.data = data, .count = count, .datatype = MPI_INT, .root = 0};
  LuchnikovETransmFrAllToOneGatherSEQ task(input);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  const auto &result = task.GetOutput();
  ASSERT_EQ(result.size(), static_cast<size_t>(count) * sizeof(int));

  const int *res_ptr = reinterpret_cast<const int *>(result.data());
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(res_ptr[i], i + 1);
  }
}

TEST(LuchnikovETransmFrAllToOneGatherMPITest, InvalidValidation) {
  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<char> data(sizeof(int));

  GatherInput input_neg_count{.data = data, .count = -1, .datatype = MPI_INT, .root = 0};
  LuchnikovETransmFrAllToOneGatherMPI task_1(input_neg_count);
  EXPECT_FALSE(task_1.Validation());

  GatherInput input_invalid_root{.data = data, .count = 1, .datatype = MPI_INT, .root = world_size};
  LuchnikovETransmFrAllToOneGatherMPI task_2(input_invalid_root);
  EXPECT_FALSE(task_2.Validation());

  GatherInput input_wrong_type{.data = data, .count = 1, .datatype = MPI_CHAR, .root = 0};
  LuchnikovETransmFrAllToOneGatherMPI task_3(input_wrong_type);
  EXPECT_FALSE(task_3.Validation());

  std::vector<char> wrong_size_data(sizeof(int) * 2);
  GatherInput input_wrong_size{.data = wrong_size_data, .count = 1, .datatype = MPI_INT, .root = 0};
  LuchnikovETransmFrAllToOneGatherMPI task_4(input_wrong_size);
  EXPECT_FALSE(task_4.Validation());
}

TEST(LuchnikovETransmFrAllToOneGatherMPITest, MiddleRootGather) {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int root = world_size / 2;
  int count = 2;
  std::vector<char> data(static_cast<size_t>(count) * sizeof(float));
  auto *d_ptr = reinterpret_cast<float *>(data.data());
  for (int i = 0; i < count; ++i) {
    d_ptr[i] = static_cast<float>(rank * count + i);
  }

  GatherInput input{.data = data, .count = count, .datatype = MPI_FLOAT, .root = root};
  LuchnikovETransmFrAllToOneGatherMPI task(input);

  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  if (rank == root) {
    const auto &result = task.GetOutput();
    ASSERT_EQ(result.size(), static_cast<size_t>(count) * static_cast<size_t>(world_size) * sizeof(float));
    const auto *res_ptr = reinterpret_cast<const float *>(result.data());
    for (int rank_id = 0; rank_id < world_size; ++rank_id) {
      for (int i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(res_ptr[(rank_id * count) + i], static_cast<float>(rank_id * count + i));
      }
    }
  }
}

TEST(LuchnikovETransmFrAllToOneGatherMPITest, LargeDataGather) {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int count = 1000;
  std::vector<char> data(static_cast<size_t>(count) * sizeof(double));
  auto *d_ptr = reinterpret_cast<double *>(data.data());
  for (int i = 0; i < count; ++i) {
    d_ptr[i] = static_cast<double>(rank * count + i);
  }

  GatherInput input{.data = data, .count = count, .datatype = MPI_DOUBLE, .root = 0};
  LuchnikovETransmFrAllToOneGatherMPI task(input);

  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  if (rank == 0) {
    const auto &result = task.GetOutput();
    EXPECT_EQ(result.size(), static_cast<size_t>(count) * static_cast<size_t>(world_size) * sizeof(double));

    const double *res_ptr = reinterpret_cast<const double *>(result.data());
    for (int i = 0; i < world_size; ++i) {
      for (int j = 0; j < count; ++j) {
        EXPECT_DOUBLE_EQ(res_ptr[i * count + j], static_cast<double>(i * count + j));
      }
    }
  }
}

TEST(LuchnikovETransmFrAllToOneGatherMPITest, NonPowerOfTwoSize) {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Проверяем, что размер не является степенью двойки
  bool is_power_of_two = (world_size > 0) && ((world_size & (world_size - 1)) == 0);

  if (!is_power_of_two && world_size > 2) {
    int count = 3;
    std::vector<char> data(static_cast<size_t>(count) * sizeof(int));
    auto *d_ptr = reinterpret_cast<int *>(data.data());
    for (int i = 0; i < count; ++i) {
      d_ptr[i] = rank * count + i;
    }

    GatherInput input{.data = data, .count = count, .datatype = MPI_INT, .root = 0};
    LuchnikovETransmFrAllToOneGatherMPI task(input);

    ASSERT_TRUE(task.Validation());
    task.PreProcessing();
    task.Run();
    task.PostProcessing();

    if (rank == 0) {
      const auto &result = task.GetOutput();
      EXPECT_EQ(result.size(), static_cast<size_t>(count) * static_cast<size_t>(world_size) * sizeof(int));

      const int *res_ptr = reinterpret_cast<const int *>(result.data());
      for (int i = 0; i < world_size; ++i) {
        for (int j = 0; j < count; ++j) {
          EXPECT_EQ(res_ptr[i * count + j], i * count + j);
        }
      }
    }
  }
}

}  // namespace

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
