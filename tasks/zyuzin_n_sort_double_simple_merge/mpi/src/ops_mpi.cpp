#include "zyuzin_n_sort_double_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "zyuzin_n_sort_double_simple_merge/common/include/common.hpp"

namespace zyuzin_n_sort_double_simple_merge {

ZyuzinNSortDoubleWithSimpleMergeMPI::ZyuzinNSortDoubleWithSimpleMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool ZyuzinNSortDoubleWithSimpleMergeMPI::ValidationImpl() {
  return true;
}

bool ZyuzinNSortDoubleWithSimpleMergeMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

void ZyuzinNSortDoubleWithSimpleMergeMPI::ConvertDoublesToBits(const int rank, const std::vector<double> &data,
                                                               std::vector<std::uint64_t> &bits_data) {
  if (rank == 0) {
    bits_data.resize(data.size(), 0);
    for (std::size_t i = 0; i < data.size(); ++i) {
      std::uint64_t x = 0;
      std::memcpy(&x, &data[i], sizeof(double));
      if ((x >> 63) == 0) {
        x ^= 0x8000000000000000;
      } else {
        x = ~x;
      }
      bits_data[i] = x;
    }
  } else {
    bits_data.resize(data.size(), 0);
  }
  MPI_Bcast(static_cast<void *>(bits_data.data()), static_cast<int>(bits_data.size()), MPI_UINT64_T, 0, MPI_COMM_WORLD);
}

std::vector<std::uint64_t> ZyuzinNSortDoubleWithSimpleMergeMPI::SortBits(const std::vector<std::uint64_t> &bits) {
  const int radix = 256;
  const std::size_t n = bits.size();
  if (n == 0) {
    return {};
  }

  std::vector<std::uint64_t> source = bits;
  std::vector<std::uint64_t> destination(n);

  for (int byte_idx = 0; byte_idx < 8; ++byte_idx) {
    int shift = byte_idx * 8;

    std::vector<std::size_t> count(radix, 0);
    for (std::uint64_t value : source) {
      std::size_t digit = (value >> shift) & 0xFF;
      ++count[digit];
    }

    for (int i = 1; i < radix; ++i) {
      count[i] += count[i - 1];
    }

    for (std::size_t i = n; i > 0; --i) {
      std::uint64_t value = source[i - 1];
      std::size_t digit = (value >> shift) & 0xFF;
      destination[--count[digit]] = value;
    }

    source.swap(destination);
  }

  return source;
}

std::vector<std::uint64_t> ZyuzinNSortDoubleWithSimpleMergeMPI::MergeSegments(
    const std::vector<std::uint64_t> &local_sorted_data, int rank, int size) {
  std::vector<std::uint64_t> current_data = local_sorted_data;

  for (int step = 1; step < size; step *= 2) {
    if ((rank / step) % 2 == 0) {
      int partner_rank = rank + step;
      if (partner_rank < size) {
        int partner_count = 0;
        MPI_Recv(&partner_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<std::uint64_t> partner_data(partner_count);
        if (partner_count > 0) {
          MPI_Recv(static_cast<void *>(partner_data.data()), partner_count, MPI_UINT64_T, partner_rank, 1,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        current_data = MergeTwoVectors(current_data, partner_data);
      }
    } else {
      int receiver_rank = rank - step;
      int send_count = static_cast<int>(current_data.size());
      MPI_Send(&send_count, 1, MPI_INT, receiver_rank, 0, MPI_COMM_WORLD);
      if (send_count > 0) {
        MPI_Send(static_cast<const void *>(current_data.data()), send_count, MPI_UINT64_T, receiver_rank, 1,
                 MPI_COMM_WORLD);
      }
      break;
    }
  }

  return current_data;
}

std::vector<std::uint64_t> ZyuzinNSortDoubleWithSimpleMergeMPI::MergeTwoVectors(const std::vector<std::uint64_t> &a,
                                                                                const std::vector<std::uint64_t> &b) {
  std::vector<std::uint64_t> result;
  result.reserve(a.size() + b.size());
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] <= b[j]) {
      result.push_back(a[i]);
      ++i;
    } else {
      result.push_back(b[j]);
      ++j;
    }
  }
  while (i < a.size()) {
    result.push_back(a[i]);
    ++i;
  }
  while (j < b.size()) {
    result.push_back(b[j]);
    ++j;
  }
  return result;
}

std::vector<double> ZyuzinNSortDoubleWithSimpleMergeMPI::ConvertBitsToDoubles(const std::vector<std::uint64_t> &data) {
  std::vector<std::uint64_t> bits = data;
  std::vector<double> doubles(data.size(), 0.0);
  for (std::size_t i = 0; i < data.size(); ++i) {
    if ((bits[i] >> 63) == 0) {
      bits[i] = ~bits[i];
    } else {
      bits[i] ^= 0x8000000000000000;
    }
    std::memcpy(&doubles[i], &bits[i], sizeof(double));
  }
  return doubles;
}

bool ZyuzinNSortDoubleWithSimpleMergeMPI::RunImpl() {
  const auto &input = GetInput();
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<std::uint64_t> bits_data;
  ConvertDoublesToBits(rank, input, bits_data);

  int bits_per_proc = static_cast<int>(bits_data.size()) / size;
  int remainder = static_cast<int>(bits_data.size()) % size;

  std::vector<int> send_bits(size, 0);
  std::vector<int> displacements(size, 0);
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    send_bits[i] = bits_per_proc + (i < remainder ? 1 : 0);
    displacements[i] = offset;
    offset += send_bits[i];
  }

  std::vector<std::uint64_t> local_bits(send_bits[rank]);
  MPI_Scatterv(static_cast<const void *>(bits_data.data()), send_bits.data(), displacements.data(), MPI_UINT64_T,
               static_cast<void *>(local_bits.data()), send_bits[rank], MPI_UINT64_T, 0, MPI_COMM_WORLD);

  std::vector<std::uint64_t> sorted_local_bits = SortBits(local_bits);

  std::vector<std::uint64_t> merged_bits = MergeSegments(sorted_local_bits, rank, size);

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(merged_bits.size());
  }
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    merged_bits.resize(total_size);
  }
  MPI_Bcast(static_cast<void *>(merged_bits.data()), total_size, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  std::vector<double> result_sorted_doubles = ConvertBitsToDoubles(merged_bits);
  GetOutput() = result_sorted_doubles;
  return true;
}

bool ZyuzinNSortDoubleWithSimpleMergeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zyuzin_n_sort_double_simple_merge
