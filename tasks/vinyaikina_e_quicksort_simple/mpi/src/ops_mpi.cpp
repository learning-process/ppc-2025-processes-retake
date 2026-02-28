#include "vinyaikina_e_quicksort_simple/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "vinyaikina_e_quicksort_simple/common/include/common.hpp"

namespace vinyaikina_e_quicksort_simple {

namespace {

std::pair<int, int> Partition(std::vector<int> &arr, int lo, int hi) {
  int pivot = arr[lo + ((hi - lo) / 2)];
  int i = lo;
  int j = hi;
  while (i <= j) {
    while (arr[i] < pivot) {
      i++;
    }
    while (arr[j] > pivot) {
      j--;
    }
    if (i <= j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    }
  }
  return {i, j};
}

void QuickSort(std::vector<int> &arr, int left, int right) {
  std::vector<std::pair<int, int>> stack;
  stack.emplace_back(left, right);
  while (!stack.empty()) {
    auto [lo, hi] = stack.back();
    stack.pop_back();
    if (lo >= hi) {
      continue;
    }
    auto [i, j] = Partition(arr, lo, hi);
    if (lo < j) {
      stack.emplace_back(lo, j);
    }
    if (i < hi) {
      stack.emplace_back(i, hi);
    }
  }
}

std::vector<int> MergeSorted(const std::vector<int> &a, const std::vector<int> &b) {
  std::vector<int> result;
  result.reserve(a.size() + b.size());
  int i = 0;
  int j = 0;
  int a_size = static_cast<int>(a.size());
  int b_size = static_cast<int>(b.size());
  while (i < a_size && j < b_size) {
    if (a[i] <= b[j]) {
      result.push_back(a[i]);
      i++;
    } else {
      result.push_back(b[j]);
      j++;
    }
  }
  while (i < a_size) {
    result.push_back(a[i]);
    i++;
  }
  while (j < b_size) {
    result.push_back(b[j]);
    j++;
  }
  return result;
}

}  // namespace

VinyaikinaEQuicksortSimpleMPI::VinyaikinaEQuicksortSimpleMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VinyaikinaEQuicksortSimpleMPI::ValidationImpl() {
  return true;
}

bool VinyaikinaEQuicksortSimpleMPI::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool VinyaikinaEQuicksortSimpleMPI::RunImpl() {
  int rank = 0;
  int proc_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  int n = static_cast<int>(data_.size());
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0) {
    return true;
  }

  std::vector<int> send_counts(proc_count);
  std::vector<int> offsets(proc_count);
  int base_chunk = n / proc_count;
  int extra = n % proc_count;
  for (int i = 0; i < proc_count; i++) {
    send_counts[i] = base_chunk + (i < extra ? 1 : 0);
    offsets[i] = (i > 0) ? (offsets[i - 1] + send_counts[i - 1]) : 0;
  }

  std::vector<int> local_data(send_counts[rank]);
  MPI_Scatterv(data_.data(), send_counts.data(), offsets.data(), MPI_INT, local_data.data(), send_counts[rank], MPI_INT,
               0, MPI_COMM_WORLD);

  if (local_data.size() > 1) {
    QuickSort(local_data, 0, static_cast<int>(local_data.size()) - 1);
  }

  std::vector<int> gathered;
  if (rank == 0) {
    gathered.resize(n);
  }
  MPI_Gatherv(local_data.data(), send_counts[rank], MPI_INT, gathered.data(), send_counts.data(), offsets.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<int> merged(gathered.begin(), gathered.begin() + send_counts[0]);
    for (int i = 1; i < proc_count; i++) {
      std::vector<int> chunk(gathered.begin() + offsets[i], gathered.begin() + offsets[i] + send_counts[i]);
      merged = MergeSorted(merged, chunk);
    }
    data_ = std::move(merged);
  }

  MPI_Bcast(data_.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool VinyaikinaEQuicksortSimpleMPI::PostProcessingImpl() {
  GetOutput() = data_;
  return true;
}

}  // namespace vinyaikina_e_quicksort_simple
