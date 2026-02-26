#include "rysev_m_max_adjacent_diff/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <utility>
#include <vector>

#include "rysev_m_max_adjacent_diff/common/include/common.hpp"

namespace rysev_m_max_adjacent_diff {

RysevMMaxAdjacentDiffMPI::RysevMMaxAdjacentDiffMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_pair(0, 0);
}

struct LocalResult {
  int diff;
  int first;
  int second;
};

LocalResult ComputeLocalMaxDiff(const std::vector<int> &data) {
  if (data.size() < 2) {
    return {-1, 0, 0};
  }
  int max_diff = std::abs(data[1] - data[0]);
  LocalResult res = {max_diff, data[0], data[1]};
  for (size_t i = 1; i < data.size() - 1; ++i) {
    int diff = std::abs(data[i + 1] - data[i]);
    if (diff > res.diff) {
      res = {diff, data[i], data[i + 1]};
    }
  }
  return res;
}

void UpdateWithBoundary(int rank, int world_size, const std::vector<int> &local_data, LocalResult &local_res) {
  int prev_last = 0;

  if (rank > 0) {
    MPI_Recv(&prev_last, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (!local_data.empty()) {
      int diff = std::abs(local_data[0] - prev_last);
      if (diff > local_res.diff) {
        local_res = {diff, prev_last, local_data[0]};
      }
    }
  }

  if (rank < world_size - 1) {
    if (!local_data.empty()) {
      MPI_Send(&local_data.back(), 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    } else {
      int dummy = 0;
      MPI_Send(&dummy, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
  }
}

bool RysevMMaxAdjacentDiffMPI::ValidationImpl() {
  return GetInput().size() >= 2;
}

bool RysevMMaxAdjacentDiffMPI::PreProcessingImpl() {
  GetOutput() = std::make_pair(0, 0);
  return true;
}

bool RysevMMaxAdjacentDiffMPI::RunImpl() {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();
  size_t n = input.size();
  if (n < 2) {
    return false;
  }

  int vec_size = static_cast<int>(n);
  MPI_Bcast(&vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> send_counts(world_size, vec_size / world_size);
  std::vector<int> displs(world_size, 0);
  int remainder = vec_size % world_size;
  for (int i = 0; i < remainder; ++i) {
    send_counts[i]++;
  }
  for (int i = 1; i < world_size; ++i) {
    displs[i] = displs[i - 1] + send_counts[i - 1];
  }

  int local_size = send_counts[rank];
  std::vector<int> local_data(local_size);
  MPI_Scatterv(const_cast<int *>(input.data()), send_counts.data(), displs.data(), MPI_INT, local_data.data(),
               local_size, MPI_INT, 0, MPI_COMM_WORLD);

  LocalResult local_res = ComputeLocalMaxDiff(local_data);

  UpdateWithBoundary(rank, world_size, local_data, local_res);

  struct {
    int diff;
    int first;
    int second;
  } send_buf = {local_res.diff, local_res.first, local_res.second};

  std::vector<decltype(send_buf)> recv_buf;
  if (rank == 0) {
    recv_buf.resize(world_size);
  }

  MPI_Gather(&send_buf, sizeof(send_buf), MPI_BYTE, recv_buf.data(), sizeof(send_buf), MPI_BYTE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    auto best = recv_buf[0];
    for (int i = 1; i < world_size; ++i) {
      if (recv_buf[i].diff > best.diff) {
        best = recv_buf[i];
      }
    }
    GetOutput() = std::make_pair(best.first, best.second);
  }
  MPI_Bcast(&GetOutput().first, 2, MPI_INT, 0, MPI_COMM_WORLD);
  return true;
}

bool RysevMMaxAdjacentDiffMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rysev_m_max_adjacent_diff
