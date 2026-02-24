#include "akhmetov_daniil_mesh_torus/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>

#include "akhmetov_daniil_mesh_torus/common/include/common.hpp"

namespace akhmetov_daniil_mesh_torus {

MeshTorusMpi::MeshTorusMpi(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

std::pair<int, int> MeshTorusMpi::ComputeGrid(int size) {
  int rows = static_cast<int>(std::sqrt(static_cast<double>(size)));

  while (rows > 1 && (size % rows != 0)) {
    --rows;
  }

  if (rows <= 0) {
    rows = 1;
  }

  int cols = size / rows;

  if (cols <= 0) {
    cols = 1;
  }

  return {rows, cols};
}

int MeshTorusMpi::RankFromCoords(int row, int col, int rows, int cols) {
  const int rr = ((row % rows) + rows) % rows;
  const int cc = ((col % cols) + cols) % cols;
  return (rr * cols) + cc;
}

std::pair<int, int> MeshTorusMpi::CoordsFromRank(int rank, int cols) {
  const int r = rank / cols;
  const int c = rank % cols;
  return {r, c};
}

std::vector<int> MeshTorusMpi::BuildPath(int rows, int cols, int source, int dest) {
  std::vector<int> path;

  if (rows <= 0 || cols <= 0) {
    path.push_back(source);
    return path;
  }

  const auto [sr, sc] = CoordsFromRank(source, cols);
  const auto [dr, dc] = CoordsFromRank(dest, cols);

  int cur_r = sr;
  int cur_c = sc;

  path.push_back(source);

  const int diff_c = dc - sc;
  const int right_dist = (diff_c >= 0) ? diff_c : diff_c + cols;
  const int left_dist = (diff_c <= 0) ? -diff_c : cols - diff_c;

  const int step_c = (right_dist <= left_dist) ? 1 : -1;
  const int steps_c = (right_dist <= left_dist) ? right_dist : left_dist;

  for (int i = 0; i < steps_c; ++i) {
    cur_c += step_c;
    path.push_back(RankFromCoords(cur_r, cur_c, rows, cols));
  }

  const int diff_r = dr - sr;
  const int down_dist = (diff_r >= 0) ? diff_r : diff_r + rows;
  const int up_dist = (diff_r <= 0) ? -diff_r : rows - diff_r;

  const int step_r = (down_dist <= up_dist) ? 1 : -1;
  const int steps_r = (down_dist <= up_dist) ? down_dist : up_dist;

  for (int i = 0; i < steps_r; ++i) {
    cur_r += step_r;
    path.push_back(RankFromCoords(cur_r, cur_c, rows, cols));
  }

  return path;
}

bool MeshTorusMpi::ValidationImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);

  if (initialized == 0) {
    return false;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  int is_valid = 0;

  if (world_rank_ == 0) {
    const auto &in = GetInput();
    if (in.source >= 0 && in.dest >= 0 && in.source < world_size_ && in.dest < world_size_) {
      is_valid = 1;
    }
  }

  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return is_valid != 0;
}

bool MeshTorusMpi::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  const auto [r, c] = ComputeGrid(world_size_);
  rows_ = r;
  cols_ = c;

  local_in_ = GetInput();
  local_out_ = OutType{};

  return true;
}

bool MeshTorusMpi::RunImpl() {
  int source = 0;
  int dest = 0;

  BroadcastSourceDest(source, dest);

  int payload_size = 0;
  BroadcastPayloadSize(source, payload_size);

  const std::vector<int> payload_buf = PreparePayloadBuffer(source, payload_size);

  const std::vector<int> path = BuildPath(rows_, cols_, source, dest);

  std::vector<int> recv_payload;

  ProcessPathCommunication(source, dest, path, payload_buf, recv_payload);

  SetOutput(dest, recv_payload, path);

  return true;
}

void MeshTorusMpi::BroadcastSourceDest(int &source, int &dest) {
  if (world_rank_ == 0) {
    const auto &in = GetInput();
    source = in.source;
    dest = in.dest;
  }

  MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&dest, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void MeshTorusMpi::BroadcastPayloadSize(int source, int &payload_size) const {
  if (world_rank_ == source) {
    payload_size = static_cast<int>(local_in_.payload.size());
  }

  MPI_Bcast(&payload_size, 1, MPI_INT, source, MPI_COMM_WORLD);
}

[[nodiscard]] std::vector<int> MeshTorusMpi::PreparePayloadBuffer(int source, int payload_size) const {
  if (world_rank_ == source && payload_size > 0) {
    return local_in_.payload;
  }

  return std::vector<int>(payload_size);
}

void MeshTorusMpi::ProcessPathCommunication(int source, int dest, const std::vector<int> &path,
                                            const std::vector<int> &payload_buf, std::vector<int> &recv_payload) const {
  if (source == dest) {
    if (world_rank_ == source) {
      recv_payload = payload_buf;
    }
    return;
  }

  const auto it = std::ranges::find(path, world_rank_);
  if (it == path.end()) {
    return;
  }

  const int my_index = static_cast<int>(std::distance(path.begin(), it));

  if (world_rank_ == source) {
    recv_payload = payload_buf;

    if (path.size() > 1) {
      const int next_rank = path[1];
      const int size_to_send = static_cast<int>(payload_buf.size());

      MPI_Send(&size_to_send, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);

      if (size_to_send > 0) {
        MPI_Send(recv_payload.data(), size_to_send, MPI_INT, next_rank, 1, MPI_COMM_WORLD);
      }
    }

    return;
  }

  const int prev_rank = path[my_index - 1];

  int recv_size = 0;
  MPI_Recv(&recv_size, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  recv_payload.resize(recv_size);

  if (recv_size > 0) {
    MPI_Recv(recv_payload.data(), recv_size, MPI_INT, prev_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (world_rank_ != dest && my_index + 1 < static_cast<int>(path.size())) {
    const int next_rank = path[my_index + 1];

    MPI_Send(&recv_size, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);

    if (recv_size > 0) {
      MPI_Send(recv_payload.data(), recv_size, MPI_INT, next_rank, 1, MPI_COMM_WORLD);
    }
  }
}

void MeshTorusMpi::SetOutput(int dest, const std::vector<int> &recv_payload, const std::vector<int> &path) {
  if (world_rank_ == dest) {
    local_out_.payload = recv_payload;
    local_out_.path = path;
    GetOutput() = local_out_;
  } else {
    GetOutput() = OutType{};
  }
}

bool MeshTorusMpi::PostProcessingImpl() {
  return true;
}

}  // namespace akhmetov_daniil_mesh_torus
