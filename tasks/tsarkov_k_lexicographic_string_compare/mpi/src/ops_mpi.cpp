#include "tsarkov_k_lexicographic_string_compare/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "tsarkov_k_lexicographic_string_compare/common/include/common.hpp"

namespace tsarkov_k_lexicographic_string_compare {

namespace {

std::vector<char> StringToBuffer(const std::string &s) {
  return {s.begin(), s.end()};
}

std::string BufferToString(const std::vector<char> &buf) {
  return {buf.begin(), buf.end()};
}

void BroadcastString(std::string *value, int root, MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  int size = 0;
  if (rank == root) {
    size = static_cast<int>(value->size());
  }

  MPI_Bcast(&size, 1, MPI_INT, root, comm);

  std::vector<char> buf;
  if (rank == root) {
    buf = StringToBuffer(*value);
  } else {
    buf.resize(static_cast<std::size_t>(size));
  }

  if (size > 0) {
    MPI_Bcast(buf.data(), size, MPI_CHAR, root, comm);
  }

  if (rank != root) {
    *value = BufferToString(buf);
  }
}

std::uint64_t FindFirstDiffLocal(const std::string &first_str, const std::string &second_str, std::size_t begin,
                                 std::size_t end) {
  const std::size_t no_index = std::numeric_limits<std::size_t>::max();
  std::size_t first_diff_index = no_index;

  for (std::size_t index = begin; index < end; ++index) {
    if (first_str[index] != second_str[index]) {
      first_diff_index = index;
      break;
    }
  }

  if (first_diff_index == no_index) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return static_cast<std::uint64_t>(first_diff_index);
}

int CompareAtIndexOrByLength(const std::string &first_str, const std::string &second_str, std::uint64_t global_first) {
  if (global_first == std::numeric_limits<std::uint64_t>::max()) {
    return (first_str.size() <= second_str.size()) ? 1 : 0;
  }

  const auto index = static_cast<std::size_t>(global_first);
  const auto first_ch = static_cast<unsigned char>(first_str[index]);
  const auto second_ch = static_cast<unsigned char>(second_str[index]);
  return (first_ch <= second_ch) ? 1 : 0;
}

}  // namespace

TsarkovKLexicographicStringCompareMPI::TsarkovKLexicographicStringCompareMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TsarkovKLexicographicStringCompareMPI::ValidationImpl() {
  return GetOutput() == 0;
}

bool TsarkovKLexicographicStringCompareMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool TsarkovKLexicographicStringCompareMPI::RunImpl() {
  int process_rank = 0;
  int process_count = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  std::string first_str;
  std::string second_str;
  if (process_rank == 0) {
    first_str = GetInput().first;
    second_str = GetInput().second;
  }

  BroadcastString(&first_str, 0, MPI_COMM_WORLD);
  BroadcastString(&second_str, 0, MPI_COMM_WORLD);

  const std::size_t min_length = std::min(first_str.size(), second_str.size());

  const std::size_t proc_cnt = (process_count > 0) ? static_cast<std::size_t>(process_count) : 1U;

  const std::size_t block_size = (min_length + proc_cnt - 1U) / proc_cnt;
  const std::size_t block_begin = std::min(min_length, block_size * static_cast<std::size_t>(process_rank));
  const std::size_t block_end = std::min(min_length, block_begin + block_size);

  const std::uint64_t local_first_diff = FindFirstDiffLocal(first_str, second_str, block_begin, block_end);

  std::uint64_t global_first_diff = std::numeric_limits<std::uint64_t>::max();
  MPI_Allreduce(&local_first_diff, &global_first_diff, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);

  const int result = CompareAtIndexOrByLength(first_str, second_str, global_first_diff);

  GetOutput() = result;
  return true;
}

bool TsarkovKLexicographicStringCompareMPI::PostProcessingImpl() {
  return (GetOutput() == 0) || (GetOutput() == 1);
}

}  // namespace tsarkov_k_lexicographic_string_compare
