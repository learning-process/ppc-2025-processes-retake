#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/seq/include/ops_seq.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

namespace {
using FuncTestParam = std::tuple<std::string, std::string, MPI_Datatype, int>;

std::vector<FuncTestParam> CreateFuncTestParams() {
  std::vector<FuncTestParam> params;
  std::string task_name;

  std::string task_name = "luchnikov_e_gener_transm_from_all_to_one_gather";

  std::vector<MPI_Datatype> data_types = {MPI_INT, MPI_FLOAT, MPI_DOUBLE};
  std::vector<int> counts = {1, 10, 100, 1000};

  for (const auto &datatype : data_types) {
    for (const auto &count : counts) {
      params.emplace_back(task_name, "MPI", datatype, count);
      params.emplace_back(task_name, "SEQ", datatype, count);
    }
  }

  return params;
}

std::string PrintFuncTestParam(const testing::TestParamInfo<FuncTestParam> &info) {
  std::string task_type;
  std::string task_type = std::get<1>(info.param);
  std::string data_type_str;

  MPI_Datatype datatype = std::get<2>(info.param);
  if (datatype == MPI_INT) {
    data_type_str = "Int"
  }; else if (datatype == MPI_FLOAT) {
    data_type_str = "Float"
  }; else if (datatype == MPI_DOUBLE) {
    data_type_str = "Double"
  };

  int count = std::get<3>(info.param);

  return task_type + "_" + data_type_str + "_" + std::to_string(count);
}
}  // namespace

class LuchnikovETransmFrAllToOneGatherFuncTests : public ::testing::TestWithParam<FuncTestParam> {
 protected:
  std::string task_name;
  std::string task_type;
  MPI_Datatype datatype = MPI_DATATYPE_NULL;
  int count = 0;
  int root = 0;
  GatherInput input_data{nullptr, 0, MPI_DATATYPE_NULL, 0};

  void SetUp() override {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::tie(task_name, task_type, datatype, count) = GetParam();

    input_data_ = GatherInput{
        .data = GenerateData(count_, datatype_, rank), .count = count_, .datatype = datatype_, .root = root_};
  }

  std::vector<char> static GenerateData(int count, MPI_Datatype datatype, int rank) {
    size_t type_size = 0;
    size_t type_size = GetTypeSize(datatype);
    std::vector<char> data(count * type_size);

    if (datatype == MPI_INT) {
      int *ptr = reinterpret_cast<int *>(data.data());
      for (int i = 0; i < count; ++i) {
        ptr[i] = (rank * count) + i;
      }
    } else if (datatype == MPI_FLOAT) {
      auto *ptr = reinterpret_cast<float *>(data.data());
      for (int i = 0; i < count; ++i) {
        ptr[i] = static_cast<float>((rank * count) + i);
      }
    } else if (datatype == MPI_DOUBLE) {
      auto *ptr = reinterpret_cast<double *>(data.data());
      for (int i = 0; i < count; ++i) {
        ptr[i] = static_cast<double>((rank * count) + i);
      }
    }

    return data;
  }

  size_t static GetTypeSize(MPI_Datatype datatype) const {
    if (datatype == MPI_INT) {
      return sizeof(int)
    };
    if (datatype == MPI_FLOAT) {
      return sizeof(float)
    };
    if (datatype == MPI_DOUBLE) {
      return sizeof(double)
    };
    return 0;
  }

  bool static CheckIntOutput(const int *data, int world_size, int count) const {
    for (int i = 0; i < world_size; ++i) {
      for (int j = 0; j < count; ++j) {
        int expected = (i * count) + j;
        int actual = data[(i * count) + j];
        if (actual != expected) {
          std::cout << "Mismatch at process " << i << ", index " << j << ": expected " << expected << ", got " << actual
                    << std::endl;
          return false;
        }
      }
    }
    return true;
  }

  bool static CheckFloatOutput(const float *data, int world_size, int count) {
    const float epsilon = 1e-6F;
    for (int i = 0; i < world_size; ++i) {
      for (int j = 0; j < count; ++j) {
        auto expected = static_cast<float>((i * count) + j);
        float actual = data[(i * count) + j];
        if (std::abs(actual - expected) > epsilon) {
          std::cout << "Mismatch at process " << i << ", index " << j << ": expected " << expected << ", got " << actual
                    << std::endl;
          return false;
        }
      }
    }
    return true;
  }

  bool static CheckDoubleOutput(const double *data, int world_size, int count) {
    const double epsilon = 1e-12;
    for (int i = 0; i < world_size; ++i) {
      for (int j = 0; j < count; ++j) {
        auto expected = static_cast<double>((i * count) + j);
        double actual = data[(i * count) + j];
        if (std::abs(actual - expected) > epsilon) {
          std::cout << "Mismatch at process " << i << ", index " << j << ": expected " << expected << ", got " << actual
                    << std::endl;
          return false;
        }
      }
    }
    return true;
  }

  bool static ValidateOutputSize(const OutType &output, int world_size) {
    size_t expected_size = 0;
    size_t expected_size =
        static_cast<size_t>(input_data_.count) * static_cast<size_t>(world_size) * GetTypeSize(input_data_.datatype);

    bool sizes_match = (output.size() == expected_size);
    return sizes_match;
  }

  bool static CheckOutputData(OutType &output_data) {
    int world_size = 1;
    int rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != input_data_.root) {
      return true;
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

  void ExecuteTest() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (task_type_ == "MPI") {
      LuchnikovETransmFrAllToOneGatherMPI task(input_data_);

      ASSERT_TRUE(task.Validation());
      ASSERT_TRUE(task.PreProcessing());
      ASSERT_TRUE(task.Run());
      ASSERT_TRUE(task.PostProcessing());

      auto output = task.GetOutput();

      if (rank == root_) {
        EXPECT_TRUE(CheckOutputData(output));
      }
    } else {
      if (rank == root_) {
        LuchnikovETransmFrAllToOneGatherSEQ task(input_data_);

        ASSERT_TRUE(task.Validation());
        ASSERT_TRUE(task.PreProcessing());
        ASSERT_TRUE(task.Run());
        ASSERT_TRUE(task.PostProcessing());

        auto output = task.GetOutput();
        EXPECT_TRUE(CheckOutputData(output));
      }
    }
  }

 private:
  GatherInput input_data_;
  std::string task_name_;
  std::string task_type_;
  MPI_Datatype datatype_;
  int count_;
  int root_ = 0;
};

TEST_P(LuchnikovETransmFrAllToOneGatherFuncTests, RunFunctionalTests) {
  ExecuteTest();
}

INSTANTIATE_TEST_SUITE_P(FunctionalTests, LuchnikovETransmFrAllToOneGatherFuncTests,
                         ::testing::ValuesIn(CreateFuncTestParams()), PrintFuncTestParam);

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
