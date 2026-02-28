#include "klimov_m_lett_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <cctype>
#include <string>
#include <algorithm>

namespace klimov_m_lett_count {

KlimovMLettCountMPI::KlimovMLettCountMPI(const InputType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs_);
}

int KlimovMLettCountMPI::CountLettersInSegment(const char *data, int length) {
  int count = 0;
  if (length <= 0) return 0;
  for (int i = 0; i < length; ++i) {
    if (std::isalpha(static_cast<unsigned char>(data[i]))) {
      ++count;
    }
  }
  return count;
}

bool KlimovMLettCountMPI::ValidationImpl() {
  return numProcs_ > 0;
}

bool KlimovMLettCountMPI::PreProcessingImpl() {
  return true;
}

bool KlimovMLettCountMPI::RunImpl() {
  int rank;
  int global_result = 0;
  std::string local_segment;
  unsigned int chunk_size = 0;
  unsigned int remainder = 0;
  unsigned int actual_len = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::string input_str = GetInput();
    chunk_size = input_str.size() / numProcs_;
    remainder = input_str.size() % numProcs_;

    MPI_Bcast(&chunk_size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&remainder, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    unsigned int start_0 = 0 * chunk_size + std::min(0U, remainder);
    unsigned int end_0 = 1 * chunk_size + std::min(1U, remainder);
    actual_len = end_0 - start_0;
    local_segment = (actual_len > 0) ? input_str.substr(start_0, actual_len) : "";

    for (int i = 1; i < numProcs_; ++i) {
      unsigned int start = i * chunk_size + std::min(static_cast<unsigned int>(i), remainder);
      unsigned int end = (i + 1) * chunk_size + std::min(static_cast<unsigned int>(i + 1), remainder);
      unsigned int seg_len = end - start;

      MPI_Send(&seg_len, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
      if (seg_len > 0) {
        MPI_Send(input_str.substr(start, seg_len).data(), static_cast<int>(seg_len), MPI_CHAR, i, 1, MPI_COMM_WORLD);
      } else {
        MPI_Send("", 0, MPI_CHAR, i, 1, MPI_COMM_WORLD);
      }
    }
  } else {
    MPI_Bcast(&chunk_size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&remainder, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Recv(&actual_len, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (actual_len > 0) {
      local_segment.resize(actual_len);
      MPI_Recv(local_segment.data(), static_cast<int>(actual_len), MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      local_segment.clear();
    }
  }

  int local_result = CountLettersInSegment(local_segment.data(), static_cast<int>(local_segment.size()));
  MPI_Reduce(&local_result, &global_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_result, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = global_result;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool KlimovMLettCountMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace klimov_m_lett_count