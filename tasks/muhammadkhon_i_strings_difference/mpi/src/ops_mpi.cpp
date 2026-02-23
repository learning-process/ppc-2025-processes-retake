#include "muhammadkhon_i_strings_difference/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "muhammadkhon_i_strings_difference/common/include/common.hpp"
namespace muhammadkhon_i_strings_difference {

MuhammadkhonIStringsDifferenceMPI::MuhammadkhonIStringsDifferenceMPI(const InType &in) {
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_size_);

  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MuhammadkhonIStringsDifferenceMPI::ValidationImpl() {
  return true;
}

bool MuhammadkhonIStringsDifferenceMPI::PreProcessingImpl() {
  return true;
}

bool MuhammadkhonIStringsDifferenceMPI::RunImpl() {
  std::string s1;
  std::string s2;
  size_t len1 = 0;
  size_t len2 = 0;

  if (proc_rank_ == 0) {
    const auto &input = GetInput();
    s1 = input.first;
    s2 = input.second;
    len1 = s1.size();
    len2 = s2.size();
  }

  MPI_Bcast(&len1, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&len2, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  const size_t min_len = std::min(len1, len2);
  const size_t max_len = std::max(len1, len2);
  const size_t base = min_len / proc_size_;
  const size_t rem = min_len % proc_size_;

  std::vector<int> send_counts(proc_size_);
  std::vector<int> displs(proc_size_);

  if (proc_rank_ == 0) {
    size_t offset = 0;
    for (int i = 0; i < proc_size_; ++i) {
      send_counts[i] = static_cast<int>(base + (std::cmp_less(i, rem) ? 1 : 0));
      displs[i] = static_cast<int>(offset);
      offset += send_counts[i];
    }
  }

  MPI_Bcast(send_counts.data(), proc_size_, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), proc_size_, MPI_INT, 0, MPI_COMM_WORLD);

  const size_t local_size = base + (std::cmp_less(proc_rank_, rem) ? 1 : 0);

  std::vector<char> local_s1(local_size);
  std::vector<char> local_s2(local_size);

  MPI_Scatterv(proc_rank_ == 0 ? const_cast<char *>(s1.data()) : nullptr, send_counts.data(), displs.data(), MPI_CHAR,
               local_s1.data(), static_cast<int>(local_size), MPI_CHAR, 0, MPI_COMM_WORLD);

  MPI_Scatterv(proc_rank_ == 0 ? const_cast<char *>(s2.data()) : nullptr, send_counts.data(), displs.data(), MPI_CHAR,
               local_s2.data(), static_cast<int>(local_size), MPI_CHAR, 0, MPI_COMM_WORLD);

  int local_diff = 0;
  for (size_t i = 0; i < local_size; ++i) {
    if (local_s1[i] != local_s2[i]) {
      local_diff++;
    }
  }

  int global_diff = 0;
  MPI_Allreduce(&local_diff, &global_diff, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = global_diff + static_cast<int>(max_len - min_len);
  return true;
}

bool MuhammadkhonIStringsDifferenceMPI::PostProcessingImpl() {
  return true;
}

}  // namespace muhammadkhon_i_strings_difference