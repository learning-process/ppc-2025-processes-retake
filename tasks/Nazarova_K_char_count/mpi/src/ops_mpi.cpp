#include "Nazarova_K_char_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>

#include "Nazarova_K_char_count/common/include/common.hpp"

namespace nazarova_k_char_count_processes {

NazarovaKCharCountMPI::NazarovaKCharCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool NazarovaKCharCountMPI::ValidationImpl() {
  return GetOutput() == 0;
}

bool NazarovaKCharCountMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool NazarovaKCharCountMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  const std::size_t n = input.text.size();

  const std::size_t start = (static_cast<std::size_t>(rank) * n) / static_cast<std::size_t>(size);
  const std::size_t end = (static_cast<std::size_t>(rank + 1) * n) / static_cast<std::size_t>(size);

  const int local_count =
      static_cast<int>(std::count(input.text.begin() + static_cast<std::ptrdiff_t>(start),
                                  input.text.begin() + static_cast<std::ptrdiff_t>(end), input.target));
  int global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_count;
  return true;
}

bool NazarovaKCharCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace nazarova_k_char_count_processes
