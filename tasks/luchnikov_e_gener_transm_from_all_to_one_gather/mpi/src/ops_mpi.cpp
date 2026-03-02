#include "luchnikov_e_gener_transm_from_all_to_one_gather/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
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

bool IsPowerOfTwo(int x) {
  return (x > 0) && ((x & (x - 1)) == 0);
}

int NextPowerOfTwo(int x) {
  int power = 1;
  while (power < x) {
    power <<= 1;
  }
  return power;
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

  const int type_size = GetTypeSize(input.datatype);
  if (type_size <= 0) {
    return false;
  }

  const size_t expected_size = static_cast<size_t>(input.count) * static_cast<size_t>(type_size);

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
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();
  const int type_size = GetTypeSize(input.datatype);
  const int block_size = input.count * type_size;

  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm temp_comm = MPI_COMM_COMM_WORLD;
  int actual_size = world_size;
  int actual_rank = rank;

  if (!IsPowerOfTwo(world_size)) {
    int next_power = NextPowerOfTwo(world_size);
    std::vector<int> ranks(next_power);
    for (int i = 0; i < next_power; ++i) {
      ranks[i] = i % world_size;
    }

    MPI_Group world_group, new_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, next_power, ranks.data(), &new_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &temp_comm);
    MPI_Group_free(&world_group);
    MPI_Group_free(&new_group);

    if (temp_comm != MPI_COMM_NULL) {
      comm = temp_comm;
      MPI_Comm_rank(comm, &actual_rank);
      MPI_Comm_size(comm, &actual_size);
    }
  }

  std::vector<char> send_buffer(block_size);
  std::copy(input.data.begin(), input.data.end(), send_buffer.begin());

  std::vector<char> recv_buffer;
  if (rank == input.root) {
    recv_buffer.resize(static_cast<size_t>(actual_size) * static_cast<size_t>(block_size));
  }

  int step = 1;
  while (step < actual_size) {
    if (actual_rank % (2 * step) == 0) {
      int source = actual_rank + step;
      if (source < actual_size) {
        int source_rank = source;
        if (!IsPowerOfTwo(world_size)) {
          source_rank = source % world_size;
        }

        int recv_size = step * block_size;
        std::vector<char> temp_recv(recv_size);
        MPI_Recv(temp_recv.data(), recv_size, MPI_BYTE, source_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        send_buffer.insert(send_buffer.end(), temp_recv.begin(), temp_recv.end());
      }
    } else {
      int dest = actual_rank - step;
      int dest_rank = dest;
      if (!IsPowerOfTwo(world_size)) {
        dest_rank = dest % world_size;
      }

      MPI_Send(send_buffer.data(), static_cast<int>(send_buffer.size()), MPI_BYTE, dest_rank, 0, MPI_COMM_WORLD);
      break;
    }
    step *= 2;
  }

  if (rank == input.root) {
    auto *out_ptr = recv_buffer.data();

    for (int i = 0; i < actual_size; ++i) {
      int source_rank = i;
      if (!IsPowerOfTwo(world_size)) {
        source_rank = i % world_size;
      }

      if (source_rank == rank) {
        std::copy(send_buffer.begin(), send_buffer.begin() + block_size, out_ptr + i * block_size);
      } else {
        bool found = false;
        for (size_t j = block_size; j < send_buffer.size(); j += block_size) {
          if (j / block_size == static_cast<size_t>(i)) {
            std::copy(send_buffer.begin() + static_cast<std::ptrdiff_t>(j),
                      send_buffer.begin() + static_cast<std::ptrdiff_t>(j + block_size), out_ptr + i * block_size);
            found = true;
            break;
          }
        }

        if (!found && i < world_size) {
          std::vector<char> direct_recv(block_size);
          MPI_Recv(direct_recv.data(), block_size, MPI_BYTE, source_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          std::copy(direct_recv.begin(), direct_recv.end(), out_ptr + i * block_size);
        }
      }
    }

    GetOutput() = std::move(recv_buffer);
  } else {
    bool sent = false;
    int temp_rank = actual_rank;
    while (temp_rank % 2 == 0 && temp_rank > 0) {
      temp_rank /= 2;
    }

    if (temp_rank > 0) {
      sent = true;
    }

    if (!sent && rank < world_size) {
      MPI_Send(send_buffer.data(), block_size, MPI_BYTE, input.root, 1, MPI_COMM_WORLD);
    }

    GetOutput() = std::vector<char>();
  }

  if (temp_comm != MPI_COMM_WORLD && temp_comm != MPI_COMM_COMM_WORLD && temp_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&temp_comm);
  }

  return true;
}

bool LuchnikovETransmFrAllToOneGatherMPI::PostProcessingImpl() {
  return true;
}

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
