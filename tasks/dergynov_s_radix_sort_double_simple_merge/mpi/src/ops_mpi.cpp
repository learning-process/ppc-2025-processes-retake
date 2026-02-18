#include "dergynov_s_radix_sort_double_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "dergynov_s_radix_sort_double_simple_merge/common/include/common.hpp"

namespace dergynov_s_radix_sort_double_simple_merge {
namespace {

void RadixSortDoubles(std::vector<double> &data) {
  if (data.size() <= 1) {
    return;
  }

  std::vector<uint64_t> keys(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    keys[i] = DoubleToSortableUint64(data[i]);
  }

  const int k_radix = 256;
  std::vector<uint64_t> temp(data.size());

  for (int shift = 0; shift < 64; shift += 8) {
    std::vector<size_t> count(k_radix + 1, 0);

    for (uint64_t key : keys) {
      uint8_t digit = (key >> shift) & 0xFF;
      ++count[digit + 1];
    }

    for (int i = 0; i < k_radix; ++i) {
      count[i + 1] += count[i];
    }

    for (uint64_t key : keys) {
      uint8_t digit = (key >> shift) & 0xFF;
      size_t pos = count[digit];
      temp[pos] = key;
      ++count[digit];
    }

    keys.swap(temp);
  }

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = SortableUint64ToDouble(keys[i]);
  }
}

std::vector<double> MergeSorted(const std::vector<double> &a, const std::vector<double> &b) {
  std::vector<double> result;
  result.reserve(a.size() + b.size());
  size_t i = 0;
  size_t j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] < b[j]) {
      result.push_back(a[i++]);
    } else {
      result.push_back(b[j++]);
    }
  }
  while (i < a.size()) {
    result.push_back(a[i++]);
  }
  while (j < b.size()) {
    result.push_back(b[j++]);
  }
  return result;
}

}  // namespace

DergynovSRadixSortDoubleSimpleMergeMPI::DergynovSRadixSortDoubleSimpleMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  result_.clear();
  std::get<1>(GetOutput()) = -1;
}

bool DergynovSRadixSortDoubleSimpleMergeMPI::ValidationImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  return initialized != 0;
}

bool DergynovSRadixSortDoubleSimpleMergeMPI::PreProcessingImpl() {
  result_.clear();
  return true;
}

bool DergynovSRadixSortDoubleSimpleMergeMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  int n = static_cast<int>(input.size());

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  int base = n / size;
  int rem = n % size;
  int offset = 0;

  for (int i = 0; i < size; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }

  std::vector<double> local_data(counts[rank]);

  MPI_Scatterv(rank == 0 ? input.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, local_data.data(),
               counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  RadixSortDoubles(local_data);

  if (rank == 0) {
    result_ = std::move(local_data);

    for (int proc = 1; proc < size; ++proc) {
      int recv_count = 0;
      MPI_Recv(&recv_count, 1, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::vector<double> part(recv_count);
      MPI_Recv(part.data(), recv_count, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      result_ = MergeSorted(result_, part);
    }
  } else {
    int send_count = static_cast<int>(local_data.size());
    MPI_Send(&send_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(local_data.data(), send_count, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
  }

  std::get<1>(GetOutput()) = rank;
  return true;
}

bool DergynovSRadixSortDoubleSimpleMergeMPI::PostProcessingImpl() {
  if (!result_.empty()) {
    std::get<0>(GetOutput()) = result_;
  }
  return true;
}

}  // namespace dergynov_s_radix_sort_double_simple_merge
