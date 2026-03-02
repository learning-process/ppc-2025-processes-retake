#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/seq/include/ops_seq.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

namespace {
using TestParam = std::tuple<std::string, std::string, int, int, MPI_Datatype, std::string>;

std::vector<TestParam> CreateTestParams() {
  std::vector<TestParam> params;

  params.emplace_back("luchnikov_e_gener_transm_from_all_to_one_gather", "MPI", 1, 0, MPI_INT, "SingleElementIntRoot0");
  params.emplace_back("luchnikov_e_gener_transm_from_all_to_one_gather", "SEQ", 1, 0, MPI_INT, "SingleElementIntRoot0");
  params.emplace_back("luchnikov_e_gener_transm_from_all_to_one_gather", "MPI", 3, 0, MPI_INT, "SmallIntRoot0");
  params.emplace_back("luchnikov_e_gener_transm_from_all_to_one_gather", "SEQ", 3, 0, MPI_INT, "SmallIntRoot0");
  params.emplace_back("luchnikov_e_gener_transm_from_all_to_one_gather", "MPI", 4, 0, MPI_FLOAT, "SmallFloatRoot0");
  params.emplace_back("luchnikov_e_gener_transm_from_all_to_one_gather", "SEQ", 4, 0, MPI_FLOAT, "SmallFloatRoot0");
  params.emplace_back("luchnikov_e_gener_transm_from_all_to_one_gather", "MPI", 3, 0, MPI_DOUBLE, "SmallDoubleRoot0");
  params.emplace_back("luchnikov_e_gener_transm_from_all_to_one_gather", "SEQ", 3, 0, MPI_DOUBLE, "SmallDoubleRoot0");
  params.emplace_back("luchnikov_e_gener_transm_from_all_to_one_gather", "MPI", 2, 1, MPI_INT, "IntRoot1");
  params.emplace_back("luchnikov_e_gener_transm_from_all_to_one_gather", "SEQ", 2, 1, MPI_INT, "IntRoot1");
  params.emplace_back("luchnikov_e_gener_transm_from_all_to_one_gather", "MPI", 2, 2, MPI_INT, "IntRoot2");
  params.emplace_back("luchnikov_e_gener_transm_from_all_to_one_gather", "SEQ", 2, 2, MPI_INT, "IntRoot2");

  return params;
}

std::string PrintTestParam(const testing::TestParamInfo<TestParam> &info) {
  return std::get<1>(info.param) + "_" + std::get<5>(info.param);
}
}  // namespace

class LuchnikovETransmFrAllToOneGatherFuncTests : public ::testing::TestWithParam<TestParam> {
 public:
  LuchnikovETransmFrAllToOneGatherFuncTests() = default;

 protected:
  void static SetUp() override {
    const auto &count = std::get<2>(GetParam());
    const auto &root = std::get<3>(GetParam());
    const auto &datatype = std::get<4>(GetParam());

    int type_size = 0;
    if (datatype == MPI_INT) {
      type_size = static_cast<int>(sizeof(int));
    } else if (datatype == MPI_FLOAT) {
      type_size = static_cast<int>(sizeof(float));
    } else if (datatype == MPI_DOUBLE) {
      type_size = static_cast<int>(sizeof(double));
    }

    const size_t total_size = static_cast<size_t>(count) * static_cast<size_t>(type_size);
    std::vector<char> data(total_size, 0);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (datatype == MPI_INT) {
      auto *data_ptr = reinterpret_cast<int *>(data.data());
      for (int i = 0; i < count; ++i) {
        data_ptr[i] = (rank * count) + i + 1;
      }
    } else if (datatype == MPI_FLOAT) {
      auto *data_ptr = reinterpret_cast<float *>(data.data());
      for (int i = 0; i < count; ++i) {
        data_ptr[i] = static_cast<float>((rank * count) + i) + 0.5F;
      }
    } else if (datatype == MPI_DOUBLE) {
      auto *data_ptr = reinterpret_cast<double *>(data.data());
      for (int i = 0; i < count; ++i) {
        data_ptr[i] = static_cast<double>((rank * count) + i) + 0.25;
      }
    }

    input_data_ = GatherInput{.data = data, .count = count, .datatype = datatype, .root = root};
  }

  static int GetTypeSize(MPI_Datatype datatype) {
    if (datatype == MPI_INT) {
      return static_cast<int>(sizeof(int));
    }
    if (datatype == MPI_FLOAT) {
      return static_cast<int>(sizeof(float));
    }
    if (datatype == MPI_DOUBLE) {
      return static_cast<int>(sizeof(double));
    }
    return 0;
  }

  bool static CheckIntOutput(const int *data, int world_size, int count) const {
    for (int i = 0; i < world_size; ++i) {
      for (int j = 0; j < count; ++j) {
        if (data[(i * count) + j] != (i * count) + j + 1) {
          return false;
        }
      }
    }
    return true;
  }

  bool static CheckFloatOutput(const float *data, int world_size, int count) const {
    for (int i = 0; i < world_size; ++i) {
      for (int j = 0; j < count; ++j) {
        if (std::abs(data[(i * count) + j] - (static_cast<float>((i * count) + j) + 0.5F)) > 1e-6F) {
          return false;
        }
      }
    }
    return true;
  }

  bool static CheckDoubleOutput(const double *data, int world_size, int count) const {
    for (int i = 0; i < world_size; ++i) {
      for (int j = 0; j < count; ++j) {
        if (std::abs(data[(i * count) + j] - (static_cast<double>((i * count) + j) + 0.25)) > 1e-12) {
          return false;
        }
      }
    }
    return true;
  }

  bool static ValidateOutputSize(const OutType &output, int world_size) const {
    if (output.empty()) {
      return false;
    }
    const size_t expected = 0;
    expected = static_cast<size_t>(input_data_.count) * static_cast<size_t>(world_size) *
               static_cast<size_t>(GetTypeSize(input_data_.datatype));
    return output.size() == expected;
  }

  bool static CheckOutputData(OutType &output_data) {
    const std::string task_type = std::get<1>(GetParam());
    const bool is_mpi = (task_type == "MPI");

    int world_size = 1;
    int rank = 0;

    if (is_mpi) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != input_data_.root) {
        return true;
      }
    }

    if (!ValidateOutputSize(output_data, world_size)) {
      return false;
    }

    if (input_data_.datatype == MPI_INT) {
      return CheckIntOutput(reinterpret_cast<const int *>(output_data.data()), world_size, input_data_.count);
    }
    if (input_data_.datatype == MPI_FLOAT) {
      return CheckFloatOutput(reinterpret_cast<const float *>(output_data.data()), world_size, input_data_.count);
    }
    if (input_data_.datatype == MPI_DOUBLE) {
      return CheckDoubleOutput(reinterpret_cast<const double *>(output_data.data()), world_size, input_data_.count);
    }

    return false;
  }

  void static ExecuteMPITask() {
    LuchnikovETransmFrAllToOneGatherMPI task(input_data_);
    ASSERT_TRUE(task.Validation());
    ASSERT_TRUE(task.PreProcessing());
    ASSERT_TRUE(task.Run());
    ASSERT_TRUE(task.PostProcessing());
    auto output = task.GetOutput();
    EXPECT_TRUE(CheckOutputData(output));
  }

  void static ExecuteSEQTask() {
    LuchnikovETransmFrAllToOneGatherSEQ task(input_data_);
    ASSERT_TRUE(task.Validation());
    ASSERT_TRUE(task.PreProcessing());
    ASSERT_TRUE(task.Run());
    ASSERT_TRUE(task.PostProcessing());
    auto output = task.GetOutput();
    EXPECT_TRUE(CheckOutputData(output));
  }

  void static ExecuteTest() {
    const std::string task_type = std::get<1>(GetParam());
    if (task_type == "MPI") {
      ExecuteMPITask();
    } else {
      ExecuteSEQTask();
    }
  }

 private:
  GatherInput input_data_{};
};

namespace {

TEST_P(LuchnikovETransmFrAllToOneGatherFuncTests, GatherCheck) {
  ExecuteTest();
}

INSTANTIATE_TEST_SUITE_P(GatherTests, LuchnikovETransmFrAllToOneGatherFuncTests,
                         ::testing::ValuesIn(CreateTestParams()), PrintTestParam);

void ValidateMPIGatherResult(const OutType &result, int count, int world_size) {
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result.size(), static_cast<size_t>(count) * static_cast<size_t>(world_size) * sizeof(int));

  const int *res_ptr = reinterpret_cast<const int *>(result.data());
  for (int i = 0; i < world_size; ++i) {
    for (int j = 0; j < count; ++j) {
      EXPECT_EQ(res_ptr[(i * count) + j], (i * count) + j + 1);
    }
  }
}

TEST(LuchnikovETransmFrAllToOneGatherMPITest, BasicMPIGather) {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int count = 3;
  std::vector<char> data(static_cast<size_t>(count) * sizeof(int), 0);

  auto *data_ptr = reinterpret_cast<int *>(data.data());
  for (int i = 0; i < count; ++i) {
    data_ptr[i] = (rank * count) + i + 1;
  }

  GatherInput input{.data = data, .count = count, .datatype = MPI_INT, .root = 0};
  LuchnikovETransmFrAllToOneGatherMPI task(input);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  if (rank == 0) {
    ValidateMPIGatherResult(task.GetOutput(), count, world_size);
  }
}

TEST(LuchnikovETransmFrAllToOneGatherSEQTest, BasicSEQGather) {
  const int count = 3;
  std::vector<char> data(static_cast<size_t>(count) * sizeof(int), 0);

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

  std::vector<char> data(sizeof(int), 0);

  GatherInput input_neg_count{.data = data, .count = -1, .datatype = MPI_INT, .root = 0};
  LuchnikovETransmFrAllToOneGatherMPI task_1(input_neg_count);
  EXPECT_FALSE(task_1.Validation());

  GatherInput input_invalid_root{.data = data, .count = 1, .datatype = MPI_INT, .root = world_size};
  LuchnikovETransmFrAllToOneGatherMPI task_2(input_invalid_root);
  EXPECT_FALSE(task_2.Validation());

  GatherInput input_wrong_type{.data = data, .count = 1, .datatype = MPI_CHAR, .root = 0};
  LuchnikovETransmFrAllToOneGatherMPI task_3(input_wrong_type);
  EXPECT_FALSE(task_3.Validation());

  std::vector<char> wrong_size_data(sizeof(int) * 2, 0);
  GatherInput input_wrong_size{.data = wrong_size_data, .count = 1, .datatype = MPI_INT, .root = 0};
  LuchnikovETransmFrAllToOneGatherMPI task_4(input_wrong_size);
  EXPECT_FALSE(task_4.Validation());
}

void ValidateFloatGatherResult(const OutType &result, int count, int world_size, int root, int rank) {
  if (rank != root) {
    return;
  }

  ASSERT_EQ(result.size(), static_cast<size_t>(count) * static_cast<size_t>(world_size) * sizeof(float));
  const auto *res_ptr = reinterpret_cast<const float *>(result.data());

  for (int rank_id = 0; rank_id < world_size; ++rank_id) {
    for (int i = 0; i < count; ++i) {
      EXPECT_FLOAT_EQ(res_ptr[(rank_id * count) + i], static_cast<float>((rank_id * count) + i));
    }
  }
}

TEST(LuchnikovETransmFrAllToOneGatherMPITest, MiddleRootGather) {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int root = world_size / 2;
  const int count = 2;
  std::vector<char> data(static_cast<size_t>(count) * sizeof(float), 0);

  auto *d_ptr = reinterpret_cast<float *>(data.data());
  for (int i = 0; i < count; ++i) {
    d_ptr[i] = static_cast<float>((rank * count) + i);
  }

  GatherInput input{.data = data, .count = count, .datatype = MPI_FLOAT, .root = root};
  LuchnikovETransmFrAllToOneGatherMPI task(input);

  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ValidateFloatGatherResult(task.GetOutput(), count, world_size, root, rank);
}

TEST(LuchnikovETransmFrAllToOneGatherMPITest, LargeDataGather) {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int count = 1000;
  std::vector<char> data(static_cast<size_t>(count) * sizeof(double), 0);

  auto *d_ptr = reinterpret_cast<double *>(data.data());
  for (int i = 0; i < count; ++i) {
    d_ptr[i] = static_cast<double>((rank * count) + i);
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

    const auto *res_ptr = reinterpret_cast<const double *>(result.data());
    for (int i = 0; i < world_size; ++i) {
      for (int j = 0; j < count; ++j) {
        EXPECT_DOUBLE_EQ(res_ptr[(i * count) + j], static_cast<double>((i * count) + j));
      }
    }
  }
}

TEST(LuchnikovETransmFrAllToOneGatherMPITest, NonPowerOfTwoSize) {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const bool is_power_of_two = (world_size > 0) && ((world_size & (world_size - 1)) == 0);

  if (is_power_of_two || world_size <= 2) {
    return;
  }

  const int count = 3;
  std::vector<char> data(static_cast<size_t>(count) * sizeof(int), 0);

  auto *d_ptr = reinterpret_cast<int *>(data.data());
  for (int i = 0; i < count; ++i) {
    d_ptr[i] = (rank * count) + i;
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
        EXPECT_EQ(res_ptr[(i * count) + j], (i * count) + j);
      }
    }
  }
}

}  // namespace
}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
