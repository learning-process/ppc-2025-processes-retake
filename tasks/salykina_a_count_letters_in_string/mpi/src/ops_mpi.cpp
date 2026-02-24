#include "salykina_a_count_letters_in_string/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <string>

#include "salykina_a_count_letters_in_string/common/include/common.hpp"

namespace salykina_a_count_letters_in_string {

SalykinaACountLettersMPI::SalykinaACountLettersMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SalykinaACountLettersMPI::ValidationImpl() {
  return !GetInput().empty() && (GetOutput() == 0);
}

bool SalykinaACountLettersMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SalykinaACountLettersMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::string &input = GetInput();
  const int string_length = static_cast<int>(input.length());

  int local_count = 0;
  int chunk_size = string_length / size;
  int remainder = string_length % size;
  int start = (rank * chunk_size) + std::min(rank, remainder);
  int end = start + chunk_size + (rank < remainder ? 1 : 0);

  for (int i = start; i < end && i < string_length; ++i) {
    if (std::isalpha(static_cast<unsigned char>(input[i])) != 0) {
      ++local_count;
    }
  }

  int total_count = 0;
  MPI_Allreduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = total_count;

  MPI_Barrier(MPI_COMM_WORLD);
  return GetOutput() >= 0;
}

bool SalykinaACountLettersMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace salykina_a_count_letters_in_string
