#include "luchnikov_e_gener_transm_from_all_to_one_gather/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
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

  // Для нестепени двойки используем простой алгоритм с прямыми передачами
  if (!IsPowerOfTwo(world_size)) {
    if (rank == input.root) {
      // Корневой процесс собирает данные от всех
      std::vector<char> recv_buffer(static_cast<size_t>(world_size) * static_cast<size_t>(block_size));

      // Копируем свои данные
      std::copy(input.data.begin(), input.data.end(),
                recv_buffer.begin() + static_cast<std::ptrdiff_t>(rank * block_size));

      // Получаем данные от всех остальных процессов
      for (int i = 0; i < world_size; ++i) {
        if (i != rank) {
          MPI_Recv(recv_buffer.data() + i * block_size, block_size, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      }

      GetOutput() = std::move(recv_buffer);
    } else {
      // Некорневые процессы отправляют свои данные корню
      MPI_Send(input.data.data(), block_size, MPI_BYTE, input.root, 0, MPI_COMM_WORLD);
      GetOutput() = std::vector<char>();
    }

    return true;
  }

  // Для степени двойки используем древовидный алгоритм
  std::vector<char> send_buffer(block_size);
  std::copy(input.data.begin(), input.data.end(), send_buffer.begin());

  std::vector<char> recv_buffer;
  if (rank == input.root) {
    recv_buffer.resize(static_cast<size_t>(world_size) * static_cast<size_t>(block_size));
    // Копируем свои данные в начало буфера
    std::copy(send_buffer.begin(), send_buffer.end(), recv_buffer.begin());
  }

  // Древовидный сбор данных
  int step = 1;
  while (step < world_size) {
    if (rank % (2 * step) == 0) {
      // Принимаем данные от процессов с шагом step
      int source = rank + step;
      if (source < world_size) {
        int recv_size = step * block_size;
        std::vector<char> temp_recv(recv_size);
        MPI_Recv(temp_recv.data(), recv_size, MPI_BYTE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Вставляем полученные данные в правильную позицию
        if (rank == input.root) {
          // Корень сразу сохраняет в финальный буфер
          std::copy(temp_recv.begin(), temp_recv.end(),
                    recv_buffer.begin() + static_cast<std::ptrdiff_t>(source * block_size));
        } else {
          // Промежуточные узлы накапливают данные для отправки выше
          send_buffer.insert(send_buffer.end(), temp_recv.begin(), temp_recv.end());
        }
      }
    } else {
      // Отправляем данные родительскому процессу
      int dest = rank - step;
      MPI_Send(send_buffer.data(), static_cast<int>(send_buffer.size()), MPI_BYTE, dest, 0, MPI_COMM_WORLD);
      break;
    }
    step *= 2;
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
