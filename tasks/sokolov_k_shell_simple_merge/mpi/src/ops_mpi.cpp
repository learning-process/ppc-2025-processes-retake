#include "sokolov_k_shell_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "sokolov_k_shell_simple_merge/common/include/common.hpp"

namespace sokolov_k_shell_simple_merge {

namespace {

void ShellSort(std::vector<int> &arr) {
  int n = static_cast<int>(arr.size());
  int gap = 1;
  while (gap < n / 3) {
    gap = (3 * gap) + 1;
  }
  while (gap >= 1) {
    for (int i = gap; i < n; i++) {
      int temp = arr[i];
      int j = i;
      while (j >= gap && arr[j - gap] > temp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = temp;
    }
    gap /= 3;
  }
}

std::vector<int> SimpleMerge(const std::vector<int> &a, const std::vector<int> &b) {
  std::vector<int> result;
  result.reserve(a.size() + b.size());
  int i = 0;
  int j = 0;
  int na = static_cast<int>(a.size());
  int nb = static_cast<int>(b.size());
  while (i < na && j < nb) {
    if (a[i] <= b[j]) {
      result.push_back(a[i]);
      i++;
    } else {
      result.push_back(b[j]);
      j++;
    }
  }
  while (i < na) {
    result.push_back(a[i]);
    i++;
  }
  while (j < nb) {
    result.push_back(b[j]);
    j++;
  }
  return result;
}

}  // namespace

SokolovKShellSimpleMergeMPI::SokolovKShellSimpleMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SokolovKShellSimpleMergeMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool SokolovKShellSimpleMergeMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    int n = GetInput();
    data_.resize(n);
    for (int i = 0; i < n; i++) {
      data_[i] = n - i;
    }
  }
  return true;
}

bool SokolovKShellSimpleMergeMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = GetInput();

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);
  int base_count = n / size;
  int remainder = n % size;

  for (int i = 0; i < size; i++) {
    send_counts[i] = base_count + (i < remainder ? 1 : 0);
    displs[i] = (i > 0) ? (displs[i - 1] + send_counts[i - 1]) : 0;
  }

  int local_size = send_counts[rank];
  std::vector<int> local_data(local_size);

  MPI_Scatterv(data_.data(), send_counts.data(), displs.data(), MPI_INT, local_data.data(), local_size, MPI_INT, 0,
               MPI_COMM_WORLD);

  ShellSort(local_data);

  MPI_Gatherv(local_data.data(), local_size, MPI_INT, data_.data(), send_counts.data(), displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<int> merged;
    for (int i = 0; i < size; i++) {
      if (send_counts[i] > 0) {
        std::vector<int> chunk(data_.begin() + displs[i], data_.begin() + displs[i] + send_counts[i]);
        merged = SimpleMerge(merged, chunk);
      }
    }
    data_ = merged;
  }

  return true;
}

bool SokolovKShellSimpleMergeMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int sorted = 1;
  if (rank == 0) {
    for (size_t i = 1; i < data_.size(); i++) {
      if (data_[i - 1] > data_[i]) {
        sorted = 0;
        break;
      }
    }
  }

  MPI_Bcast(&sorted, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (sorted == 0) {
    return false;
  }

  GetOutput() = GetInput();
  return true;
}

}  // namespace sokolov_k_shell_simple_merge
