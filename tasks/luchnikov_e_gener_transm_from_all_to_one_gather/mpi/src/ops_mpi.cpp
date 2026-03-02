#include "luchnikov_e_gener_transm_from_all_to_one_gather/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

namespace {
int GetTypeSize(MPI_Datatype datatype) {
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
}  // namespace

LuchnikovETransmFrAllToOneGatherMPI::LuchnikovETransmFrAllToOneGatherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool LuchnikovETransmFrAllToOneGatherMPI::ValidationImpl() {
  const auto &input = GetInput();

  if (input.count <= 0) {
    return false;
  }

  if (input.datatype != MPI_INT && input.datatype != MPI_FLOAT && input.datatype != MPI_DOUBLE) {
    return false;
  }

  if (input.root < 0) {
    return false;
  }

  int type_size = GetTypeSize(input.datatype);
  if (type_size <= 0) {
    return false;
  }

  size_t expected_size = (size_t)input.count * (size_t)type_size;
  if (input.data.size() != expected_size) {
    return false;
  }

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  return input.root < size;
}

bool LuchnikovETransmFrAllToOneGatherMPI::PreProcessingImpl() {
  return true;
}

bool LuchnikovETransmFrAllToOneGatherMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();

  int type_size = GetTypeSize(input.datatype);
  
  int block_size = input.count * type_size;

  std::vector<char> recv_buffer;
  if (rank == input.root) {
    recv_buffer.resize((size_t)input.count * (size_t)size * (size_t)type_size);
  }

  if (rank == input.root) {
    char* out_ptr = recv_buffer.data();
    std::copy(input.data.begin(), input.data.end(), out_ptr + rank * block_size);
    
    for (int i = 0; i < size; i++) {
      if (i != rank) {
        MPI_Recv(out_ptr + i * block_size, block_size, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  } else {
    MPI_Send(input.data.data(), block_size, MPI_BYTE, input.root, 0, MPI_COMM_WORLD);
  }

  if (rank == input.root) {
    GetOutput() = std::move(recv_buffer);
  } else {
    GetOutput() = std::vector<char>();
  }

  return true;
}

bool LuchnikovETransmFrAllToOneGatherMPI::PostProcessingImpl() {
  return true;
}

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather