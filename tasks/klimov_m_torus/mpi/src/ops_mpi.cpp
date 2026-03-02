#include "klimov_m_torus/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>

#include "klimov_m_torus/common/include/common.hpp"

namespace klimov_m_torus {

TorusMeshCommunicator::TorusMeshCommunicator(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

std::pair<int, int> TorusMeshCommunicator::CalculateGridSize(int totalProcesses) {
  int rows = static_cast<int>(std::sqrt(static_cast<double>(totalProcesses)));
  while (rows > 1 && (totalProcesses % rows != 0)) {
    --rows;
  }
  if (rows <= 0) {
    rows = 1;
  }
  int cols = totalProcesses / rows;
  if (cols <= 0) {
    cols = 1;
  }
  return {rows, cols};
}

int TorusMeshCommunicator::CombineCoordinates(int row, int col, int rows, int cols) {
  int wrapped_row = ((row % rows) + rows) % rows;
  int wrapped_col = ((col % cols) + cols) % cols;
  return (wrapped_row * cols) + wrapped_col;
}

std::pair<int, int> TorusMeshCommunicator::SplitRank(int rank, int cols) {
  int r = rank / cols;
  int c = rank % cols;
  return {r, c};
}

std::vector<int> TorusMeshCommunicator::BuildMessageRoute(int rows, int cols, int from, int to) {
  std::vector<int> route;
  if (rows <= 0 || cols <= 0) {
    route.push_back(from);
    return route;
  }

  auto [src_row, src_col] = SplitRank(from, cols);
  auto [dst_row, dst_col] = SplitRank(to, cols);

  int cur_row = src_row;
  int cur_col = src_col;
  route.push_back(from);

  int col_diff = dst_col - src_col;
  int right_steps = (col_diff >= 0) ? col_diff : col_diff + cols;
  int left_steps = (col_diff <= 0) ? -col_diff : cols - col_diff;
  int col_step = (right_steps <= left_steps) ? 1 : -1;
  int col_moves = (right_steps <= left_steps) ? right_steps : left_steps;

  for (int i = 0; i < col_moves; ++i) {
    cur_col += col_step;
    route.push_back(CombineCoordinates(cur_row, cur_col, rows, cols));
  }

  int row_diff = dst_row - src_row;
  int down_steps = (row_diff >= 0) ? row_diff : row_diff + rows;
  int up_steps = (row_diff <= 0) ? -row_diff : rows - row_diff;
  int row_step = (down_steps <= up_steps) ? 1 : -1;
  int row_moves = (down_steps <= up_steps) ? down_steps : up_steps;

  for (int i = 0; i < row_moves; ++i) {
    cur_row += row_step;
    route.push_back(CombineCoordinates(cur_row, cur_col, rows, cols));
  }

  return route;
}

bool TorusMeshCommunicator::ValidationImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    return false;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &current_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &total_ranks_);

  int valid = 0;
  if (current_rank_ == 0) {
    const auto &req = GetInput();
    if (req.sender >= 0 && req.receiver >= 0 && req.sender < total_ranks_ && req.receiver < total_ranks_) {
      valid = 1;
    }
  }
  MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid != 0;
}

bool TorusMeshCommunicator::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &current_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &total_ranks_);

  auto [r, c] = CalculateGridSize(total_ranks_);
  grid_rows_ = r;
  grid_cols_ = c;

  local_request_ = GetInput();
  local_response_ = OutType{};
  return true;
}

bool TorusMeshCommunicator::RunImpl() {
  int sender = 0, receiver = 0;
  DistributeSenderReceiver(sender, receiver);

  int data_len = 0;
  DistributeDataLength(sender, data_len);

  std::vector<int> send_buffer = AssembleSendBuffer(sender, data_len);
  std::vector<int> message_route = BuildMessageRoute(grid_rows_, grid_cols_, sender, receiver);

  std::vector<int> received_data;
  RelayMessage(sender, receiver, message_route, send_buffer, received_data);

  SaveFinalResult(receiver, received_data, message_route);
  return true;
}

void TorusMeshCommunicator::DistributeSenderReceiver(int &src, int &dst) {
  if (current_rank_ == 0) {
    const auto &req = GetInput();
    src = req.sender;
    dst = req.receiver;
  }
  MPI_Bcast(&src, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&dst, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void TorusMeshCommunicator::DistributeDataLength(int src, int &len) const {
  if (current_rank_ == src) {
    len = static_cast<int>(local_request_.data.size());
  }
  MPI_Bcast(&len, 1, MPI_INT, src, MPI_COMM_WORLD);
}

std::vector<int> TorusMeshCommunicator::AssembleSendBuffer(int src, int len) const {
  std::vector<int> buffer(len);
  if (current_rank_ == src && len > 0) {
    std::copy(local_request_.data.begin(), local_request_.data.end(), buffer.begin());
  }
  return buffer;
}

void TorusMeshCommunicator::RelayMessage(int src, int dst, const std::vector<int> &route,
                                         const std::vector<int> &buffer, std::vector<int> &output) const {
  const int route_len = static_cast<int>(route.size());
  auto it = std::find(route.begin(), route.end(), current_rank_);
  bool on_route = (it != route.end());
  int my_pos = on_route ? static_cast<int>(std::distance(route.begin(), it)) : -1;

  if (src == dst) {
    if (current_rank_ == src) {
      output = buffer;
    }
  } else if (current_rank_ == src) {
    output = buffer;
    if (route_len > 1) {
      int next_hop = route[1];
      int send_len = static_cast<int>(buffer.size());
      MPI_Send(&send_len, 1, MPI_INT, next_hop, 0, MPI_COMM_WORLD);
      if (send_len > 0) {
        MPI_Send(output.data(), send_len, MPI_INT, next_hop, 1, MPI_COMM_WORLD);
      }
    }
  } else if (on_route) {
    int prev_hop = route[my_pos - 1];
    int recv_len = 0;
    MPI_Recv(&recv_len, 1, MPI_INT, prev_hop, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    output.resize(recv_len);
    if (recv_len > 0) {
      MPI_Recv(output.data(), recv_len, MPI_INT, prev_hop, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (current_rank_ != dst && my_pos + 1 < route_len) {
      int next_hop = route[my_pos + 1];
      MPI_Send(&recv_len, 1, MPI_INT, next_hop, 0, MPI_COMM_WORLD);
      if (recv_len > 0) {
        MPI_Send(output.data(), recv_len, MPI_INT, next_hop, 1, MPI_COMM_WORLD);
      }
    }
  }
}

void TorusMeshCommunicator::SaveFinalResult(int dst, const std::vector<int> &output, const std::vector<int> &route) {
  if (current_rank_ == dst) {
    local_response_.received_data = output;
    local_response_.route = route;
    GetOutput() = local_response_;
  } else {
    GetOutput() = OutType{};
  }
}

bool TorusMeshCommunicator::PostProcessingImpl() {
  return true;
}

}  // namespace klimov_m_torus
