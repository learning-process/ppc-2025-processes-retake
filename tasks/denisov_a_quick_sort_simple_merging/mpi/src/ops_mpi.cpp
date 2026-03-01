#include "denisov_a_quick_sort_simple_merging/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "denisov_a_quick_sort_simple_merging/common/include/common.hpp"

namespace denisov_a_quick_sort_simple_merging {

DenisovAQuickSortMergeMPI::DenisovAQuickSortMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool DenisovAQuickSortMergeMPI::ValidationImpl() {
  return GetOutput().empty();
}

bool DenisovAQuickSortMergeMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto &out = GetOutput();
  const auto &in = GetInput();

  out = in;

  if (rank == 0) {
    if (out.size() != in.size()) {
      return false;
    }
    for (size_t i = 0; i < out.size(); i++) {
      if (out[i] != in[i]) {
        return false;
      }
    }
  }

  return true;
}

bool DenisovAQuickSortMergeMPI::RunImpl() {
  int rank = 0;
  int world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  auto &global = GetOutput();

  if (global.empty()) {
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  int total = static_cast<int>(global.size());
  std::vector<int> parts(world);
  std::vector<int> shifts(world);

  int chunk = total / world;
  int extra = total % world;

  for (int i = 0; i < world; i++) {
    parts[i] = chunk + (i < extra ? 1 : 0);
    shifts[i] = (i == 0 ? 0 : shifts[i - 1] + parts[i - 1]);
  }

  std::vector<int> local(parts[rank]);

  MPI_Scatterv(rank == 0 ? global.data() : nullptr, parts.data(), shifts.data(), MPI_INT, local.data(), parts[rank],
               MPI_INT, 0, MPI_COMM_WORLD);

  if (!local.empty()) {
    QuickSort(local, 0, static_cast<int>(local.size()) - 1);
  }

  if (rank == 0) {
    global = local;

    for (int i = 1; i < world; i++) {
      std::vector<int> recv(parts[i]);
      MPI_Recv(recv.data(), parts[i], MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      global = Merge(global, recv);
    }

    for (int i = 1; i < world; i++) {
      MPI_Send(global.data(), static_cast<int>(global.size()), MPI_INT, i, 1, MPI_COMM_WORLD);
    }

  } else {
    MPI_Send(local.data(), static_cast<int>(local.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);

    MPI_Status st;
    int recv_len = 0;
    MPI_Probe(0, 1, MPI_COMM_WORLD, &st);
    MPI_Get_count(&st, MPI_INT, &recv_len);

    global.resize(recv_len);
    MPI_Recv(global.data(), recv_len, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool DenisovAQuickSortMergeMPI::PostProcessingImpl() {
  const auto &out = GetOutput();
  const auto &in = GetInput();

  if (out.empty()) {
    return in.empty();
  }

  if (!std::ranges::is_sorted(out)) {
    return false;
  }

  if (out.size() != in.size()) {
    return false;
  }

  int64_t sum_in = 0;
  int64_t sum_out = 0;

  for (int v : in) {
    sum_in += v;
  }
  for (int v : out) {
    sum_out += v;
  }

  return sum_in == sum_out;
}

namespace {

inline int Partition(std::vector<int> &data, int left, int right) {
  int pivot = data[(left + right) / 2];
  int i = left;
  int j = right;

  while (i <= j) {
    while (data[i] < pivot) {
      i++;
    }
    while (data[j] > pivot) {
      j--;
    }

    if (i <= j) {
      std::swap(data[i], data[j]);
      i++;
      j--;
    }
  }
  return i;
}

inline void PushRange(std::vector<std::pair<int, int>> &stack, int l, int r) {
  if (l < r) {
    stack.emplace_back(l, r);
  }
}

}  // namespace

void DenisovAQuickSortMergeMPI::QuickSort(std::vector<int> &data, int begin, int end) {
  std::vector<std::pair<int, int>> stack;
  PushRange(stack, begin, end);

  while (!stack.empty()) {
    auto [l, r] = stack.back();
    stack.pop_back();

    int mid = Partition(data, l, r);

    if (mid - 1 - l < r - mid) {
      PushRange(stack, l, mid - 1);
      l = mid;
    } else {
      PushRange(stack, mid, r);
      r = mid - 1;
    }

    if (l < r) {
      stack.emplace_back(l, r);
    }
  }
}

std::vector<int> DenisovAQuickSortMergeMPI::Merge(const std::vector<int> &left_block,
                                                  const std::vector<int> &right_block) {
  std::vector<int> res;
  res.reserve(left_block.size() + right_block.size());

  size_t i = 0;
  size_t j = 0;

  while (i < left_block.size() && j < right_block.size()) {
    if (left_block[i] <= right_block[j]) {
      res.push_back(left_block[i]);
      i++;
    } else {
      res.push_back(right_block[j]);
      j++;
    }
  }

  while (i < left_block.size()) {
    res.push_back(left_block[i]);
    i++;
  }

  while (j < right_block.size()) {
    res.push_back(right_block[j]);
    j++;
  }

  return res;
}

}  // namespace denisov_a_quick_sort_simple_merging
