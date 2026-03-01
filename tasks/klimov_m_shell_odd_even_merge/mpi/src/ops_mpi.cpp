#include "klimov_m_shell_odd_even_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace klimov_m_shell_odd_even_merge {

ShellBatcherMPI::ShellBatcherMPI(const InputType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
}

bool ShellBatcherMPI::ValidationImpl() { return !GetInput().empty(); }
bool ShellBatcherMPI::PreProcessingImpl() { return true; }
bool ShellBatcherMPI::PostProcessingImpl() { return true; }

void ShellSortLocal(std::vector<int> &data) {
  const size_t len = data.size();
  for (size_t gap = len / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < len; ++i) {
      int tmp = data[i];
      size_t j = i;
      while (j >= gap && data[j - gap] > tmp) {
        data[j] = data[j - gap];
        j -= gap;
      }
      data[j] = tmp;
    }
  }
}

std::vector<int> MergeEvenLeft(std::vector<int> &left, std::vector<int> &right,
                               int chunk, int rank, MPI_Comm comm) {
  std::vector<int> left_even, right_even;
  left_even.reserve(chunk);
  right_even.reserve(chunk);

  for (int k = 0; k < chunk; k += 2) {
    left_even.push_back(left[k]);
    right_even.push_back(right[k]);
  }

  std::vector<int> merged_even(left_even.size() + right_even.size());
  std::merge(left_even.begin(), left_even.end(),
             right_even.begin(), right_even.end(),
             merged_even.begin());

  std::vector<int> recv_odd((2 * static_cast<size_t>(chunk)) - merged_even.size());
  MPI_Sendrecv(merged_even.data(), static_cast<int>(merged_even.size()), MPI_INT, rank + 1, 0,
               recv_odd.data(), static_cast<int>(recv_odd.size()), MPI_INT, rank + 1, 0,
               comm, MPI_STATUS_IGNORE);

  std::vector<int> result;
  result.reserve(chunk);
  if (chunk > 0) result.push_back(merged_even[0]);

  int idx = 0;
  while (result.size() < static_cast<size_t>(chunk)) {
    int v1 = merged_even[idx + 1];
    int v2 = recv_odd[idx];
    if (v1 > v2) std::swap(v1, v2);

    result.push_back(v1);
    if (result.size() < static_cast<size_t>(chunk)) {
      result.push_back(v2);
    }
    ++idx;
  }
  return result;
}

std::vector<int> MergeOddRight(std::vector<int> &left, std::vector<int> &right,
                               int chunk, int rank, MPI_Comm comm) {
  std::vector<int> left_odd, right_odd;
  left_odd.reserve(chunk);
  right_odd.reserve(chunk);

  for (int k = 1; k < chunk; k += 2) {
    left_odd.push_back(left[k]);
    right_odd.push_back(right[k]);
  }

  std::vector<int> merged_odd(left_odd.size() + right_odd.size());
  std::merge(left_odd.begin(), left_odd.end(),
             right_odd.begin(), right_odd.end(),
             merged_odd.begin());

  std::vector<int> recv_even((2 * static_cast<size_t>(chunk)) - merged_odd.size());
  MPI_Sendrecv(merged_odd.data(), static_cast<int>(merged_odd.size()), MPI_INT, rank - 1, 0,
               recv_even.data(), static_cast<int>(recv_even.size()), MPI_INT, rank - 1, 0,
               comm, MPI_STATUS_IGNORE);

  std::vector<int> result;
  result.reserve(chunk);
  size_t idx = (chunk % 2 == 0) ? (static_cast<size_t>(chunk) / 2) - 1
                                 : (static_cast<size_t>(chunk) - 1) / 2;

  if (chunk % 2 == 0) {
    int v1 = recv_even[idx + 1];
    int v2 = merged_odd[idx];
    if (v1 > v2) std::swap(v1, v2);
    result.push_back(v2);
    ++idx;
  }

  for (; idx + 1 < recv_even.size() && idx < merged_odd.size(); ++idx) {
    int v1 = recv_even[idx + 1];
    int v2 = merged_odd[idx];
    if (v1 > v2) std::swap(v1, v2);
    result.push_back(v1);
    result.push_back(v2);
  }

  while (idx + 1 < recv_even.size()) result.push_back(recv_even[++idx]);
  while (idx < merged_odd.size()) result.push_back(merged_odd[idx++]);

  return result;
}

void ExchangeWithRight(int rank, std::vector<int> &chunk, int chunk_size, MPI_Comm comm) {
  std::vector<int> neighbor(chunk_size);
  MPI_Sendrecv(chunk.data(), chunk_size, MPI_INT, rank + 1, 0,
               neighbor.data(), chunk_size, MPI_INT, rank + 1, 0,
               comm, MPI_STATUS_IGNORE);
  chunk = MergeEvenLeft(chunk, neighbor, chunk_size, rank, comm);
}

void ExchangeWithLeft(int rank, std::vector<int> &chunk, int chunk_size, MPI_Comm comm) {
  std::vector<int> neighbor(chunk_size);
  MPI_Sendrecv(chunk.data(), chunk_size, MPI_INT, rank - 1, 0,
               neighbor.data(), chunk_size, MPI_INT, rank - 1, 0,
               comm, MPI_STATUS_IGNORE);
  chunk = MergeOddRight(neighbor, chunk, chunk_size, rank, comm);
}

void EvenStep(int rank, int procs, std::vector<int> &chunk, int chunk_size, MPI_Comm comm) {
  if (procs % 2 != 0 && rank == procs - 1) return;
  if (rank % 2 == 0) {
    ExchangeWithRight(rank, chunk, chunk_size, comm);
  } else {
    ExchangeWithLeft(rank, chunk, chunk_size, comm);
  }
}

void OddStep(int rank, int procs, std::vector<int> &chunk, int chunk_size, MPI_Comm comm) {
  if (rank == 0 || (procs % 2 == 0 && rank == procs - 1)) return;
  if (rank % 2 == 0) {
    ExchangeWithLeft(rank, chunk, chunk_size, comm);
  } else {
    ExchangeWithRight(rank, chunk, chunk_size, comm);
  }
}

bool ShellBatcherMPI::RunImpl() {
  int procs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> global_data;
  int total_elements = 0;

  if (rank == 0) {
    global_data = GetInput();
    total_elements = static_cast<int>(global_data.size());
  }
  MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int padding = (total_elements % procs == 0) ? 0 : (procs - (total_elements % procs));
  if (rank == 0) {
    global_data.insert(global_data.end(), padding, std::numeric_limits<int>::max());
  }

  int local_size = (total_elements + padding) / procs;
  std::vector<int> local_chunk(local_size);
  MPI_Scatter(global_data.data(), local_size, MPI_INT,
              local_chunk.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

  ShellSortLocal(local_chunk);

  for (int step = 0; step <= procs; ++step) {
    if (step % 2 == 0) {
      EvenStep(rank, procs, local_chunk, local_size, MPI_COMM_WORLD);
    } else {
      OddStep(rank, procs, local_chunk, local_size, MPI_COMM_WORLD);
    }
  }

  global_data.resize(total_elements + padding);
  MPI_Allgather(local_chunk.data(), local_size, MPI_INT,
                global_data.data(), local_size, MPI_INT, MPI_COMM_WORLD);

  global_data.resize(total_elements);
  GetOutput() = global_data;
  return true;
}

}  // namespace klimov_m_shell_odd_even_merge