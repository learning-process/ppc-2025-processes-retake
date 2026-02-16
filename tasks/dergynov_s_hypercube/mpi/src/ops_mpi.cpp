#include "dergynov_s_hypercube/mpi/include/ops_mpi.hpp"
#include "dergynov_s_hypercube/common/include/common.hpp"

#include <mpi.h>

#include <cstddef>
#include <numeric>
#include <vector>

namespace dergynov_s_hypercube {

DergynovSHypercubeMPI::DergynovSHypercubeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

int DergynovSHypercubeMPI::CalcDim(int size) {
  int dim = 0;
  int cap = 1;
  while (cap < size) {
    cap <<= 1;
    dim++;
  }
  return dim;
}

std::vector<int> DergynovSHypercubeMPI::BuildPath(int src, int dst, int dim) {
  std::vector<int> path;
  int cur = src;
  int diff = src ^ dst;
  path.push_back(cur);

  for (int dim_idx = 0; dim_idx < dim; ++dim_idx) {
    if ((diff & (1 << dim_idx)) != 0) {
      cur ^= (1 << dim_idx);
      path.push_back(cur);
      if (cur == dst) {
        break;
      }
    }
  }
  return path;
}

void DergynovSHypercubeMPI::FindPos(int rank, const std::vector<int> &path, int &pos, int &next, int &prev) {
  pos = -1;
  next = -1;
  prev = -1;
  for (std::size_t idx = 0; idx < path.size(); ++idx) {
    if (path[idx] == rank) {
      pos = static_cast<int>(idx);
      if (idx > 0) {
        prev = path[idx - 1];
      }
      if (idx + 1 < path.size()) {
        next = path[idx + 1];
      }
      break;
    }
  }
}

void DergynovSHypercubeMPI::SendVec(const std::vector<int> &data, int to) {
  int sz = static_cast<int>(data.size());
  MPI_Send(&sz, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
  if (sz > 0) {
    MPI_Send(data.data(), sz, MPI_INT, to, 1, MPI_COMM_WORLD);
  }
}

void DergynovSHypercubeMPI::RecvVec(std::vector<int> &data, int from) {
  int sz = 0;
  MPI_Recv(&sz, 1, MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  data.resize(sz);
  if (sz > 0) {
    MPI_Recv(data.data(), sz, MPI_INT, from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void DergynovSHypercubeMPI::BusyWork(int iters) {
  volatile int x = 0;
  for (int i = 0; i < iters; ++i) {
    x = (x * 31 + i) % 1000003;
  }
  (void)x;
}

bool DergynovSHypercubeMPI::ValidationImpl() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto &in = GetInput();
  if (in[0] < 0 || in[0] >= size) {
    in[0] = 0;
  }
  if (in[1] < 0 || in[1] >= size) {
    in[1] = size - 1;
  }
  return true;
}

bool DergynovSHypercubeMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool DergynovSHypercubeMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &in = GetInput();
  int src = in[0];
  int dst = in[1];
  int data_size = in[2];

  std::vector<int> data;
  if (rank == src) {
    data.resize(data_size, 1);
  }

  if (src == dst) {
    MPI_Bcast(&data_size, 1, MPI_INT, dst, MPI_COMM_WORLD);
    if (rank != dst) {
      data.resize(data_size);
    }
    MPI_Bcast(data.data(), data_size, MPI_INT, dst, MPI_COMM_WORLD);
    GetOutput() = std::accumulate(data.begin(), data.end(), 0);
    return true;
  }

  int dim = CalcDim(size);
  auto path = BuildPath(src, dst, dim);

  int pos = -1;
  int next = -1;
  int prev = -1;
  FindPos(rank, path, pos, next, prev);

  if (pos != -1) {
    if (rank == src) {
      BusyWork(120000);
      SendVec(data, next);
    } else if (rank == dst) {
      RecvVec(data, prev);
      BusyWork(120000);
    } else {
      RecvVec(data, prev);
      BusyWork(120000);
      SendVec(data, next);
    }
  }

  int final_size = static_cast<int>(data.size());
  MPI_Bcast(&final_size, 1, MPI_INT, dst, MPI_COMM_WORLD);
  if (pos == -1) {
    data.resize(final_size);
  }
  MPI_Bcast(data.data(), final_size, MPI_INT, dst, MPI_COMM_WORLD);

  GetOutput() = std::accumulate(data.begin(), data.end(), 0);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool DergynovSHypercubeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dergynov_s_hypercube