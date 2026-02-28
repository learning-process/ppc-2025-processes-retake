#include "klimov_m_torus/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>

#include "klimov_m_torus/common/include/common.hpp"

namespace klimov_m_torus {

TorusNetworkMpi::TorusNetworkMpi(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

std::pair<int, int> TorusNetworkMpi::CalculateGridDimensions(int totalProcs) {
  int rows = static_cast<int>(std::sqrt(static_cast<double>(totalProcs)));
  while (rows > 1 && (totalProcs % rows != 0)) {
    --rows;
  }
  if (rows <= 0) rows = 1;
  int cols = totalProcs / rows;
  if (cols <= 0) cols = 1;
  return {rows, cols};
}

int TorusNetworkMpi::RankFromCoordinates(int row, int col, int rows, int cols) {
  int wrappedRow = ((row % rows) + rows) % rows;
  int wrappedCol = ((col % cols) + cols) % cols;
  return (wrappedRow * cols) + wrappedCol;
}

std::pair<int, int> TorusNetworkMpi::CoordinatesFromRank(int rank, int cols) {
  int r = rank / cols;
  int c = rank % cols;
  return {r, c};
}

std::vector<int> TorusNetworkMpi::BuildRoute(int rows, int cols, int from, int to) {
  std::vector<int> route;
  if (rows <= 0 || cols <= 0) {
    route.push_back(from);
    return route;
  }

  auto [srcRow, srcCol] = CoordinatesFromRank(from, cols);
  auto [dstRow, dstCol] = CoordinatesFromRank(to, cols);

  int curRow = srcRow;
  int curCol = srcCol;
  route.push_back(from);

  int colDiff = dstCol - srcCol;
  int rightSteps = (colDiff >= 0) ? colDiff : colDiff + cols;
  int leftSteps = (colDiff <= 0) ? -colDiff : cols - colDiff;
  int stepCol = (rightSteps <= leftSteps) ? 1 : -1;
  int stepsCol = (rightSteps <= leftSteps) ? rightSteps : leftSteps;

  for (int i = 0; i < stepsCol; ++i) {
    curCol += stepCol;
    route.push_back(RankFromCoordinates(curRow, curCol, rows, cols));
  }

  int rowDiff = dstRow - srcRow;
  int downSteps = (rowDiff >= 0) ? rowDiff : rowDiff + rows;
  int upSteps = (rowDiff <= 0) ? -rowDiff : rows - rowDiff;
  int stepRow = (downSteps <= upSteps) ? 1 : -1;
  int stepsRow = (downSteps <= upSteps) ? downSteps : upSteps;

  for (int i = 0; i < stepsRow; ++i) {
    curRow += stepRow;
    route.push_back(RankFromCoordinates(curRow, curCol, rows, cols));
  }

  return route;
}

bool TorusNetworkMpi::ValidationImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) return false;

  MPI_Comm_rank(MPI_COMM_WORLD, &currentRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);

  int valid = 0;
  if (currentRank_ == 0) {
    const auto &req = GetInput();
    if (req.source >= 0 && req.dest >= 0 &&
        req.source < worldSize_ && req.dest < worldSize_) {
      valid = 1;
    }
  }
  MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid != 0;
}

bool TorusNetworkMpi::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &currentRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);

  auto [r, c] = CalculateGridDimensions(worldSize_);
  gridRows_ = r;
  gridCols_ = c;

  localInput_ = GetInput();
  localOutput_ = OutType{};
  return true;
}

bool TorusNetworkMpi::RunImpl() {
  int sender = 0, receiver = 0;
  BroadcastSourceAndDestination(sender, receiver);

  int dataSize = 0;
  BroadcastDataSize(sender, dataSize);

  std::vector<int> dataBuffer = PrepareDataBuffer(sender, dataSize);
  std::vector<int> transmissionRoute = BuildRoute(gridRows_, gridCols_, sender, receiver);

  std::vector<int> receivedData;
  ForwardData(sender, receiver, transmissionRoute, dataBuffer, receivedData);

  SaveResult(receiver, receivedData, transmissionRoute);
  return true;
}

void TorusNetworkMpi::BroadcastSourceAndDestination(int &src, int &dst) {
  if (currentRank_ == 0) {
    const auto &req = GetInput();
    src = req.source;
    dst = req.dest;
  }
  MPI_Bcast(&src, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&dst, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void TorusNetworkMpi::BroadcastDataSize(int src, int &dataSize) const {
  if (currentRank_ == src) {
    dataSize = static_cast<int>(localInput_.payload.size());
  }
  MPI_Bcast(&dataSize, 1, MPI_INT, src, MPI_COMM_WORLD);
}

std::vector<int> TorusNetworkMpi::PrepareDataBuffer(int src, int dataSize) const {
  std::vector<int> buffer(dataSize);
  if (currentRank_ == src && dataSize > 0) {
    std::copy(localInput_.payload.begin(), localInput_.payload.end(), buffer.begin());
  }
  return buffer;
}

void TorusNetworkMpi::ForwardData(int src, int dst, const std::vector<int> &route,
                                   const std::vector<int> &buffer, std::vector<int> &received) const {
  const int routeLen = static_cast<int>(route.size());
  auto it = std::find(route.begin(), route.end(), currentRank_);
  bool onRoute = (it != route.end());
  int myIndex = onRoute ? static_cast<int>(std::distance(route.begin(), it)) : -1;

  if (src == dst) {
    if (currentRank_ == src) {
      received = buffer;
    }
  } else if (currentRank_ == src) {
    received = buffer;
    if (routeLen > 1) {
      int nextHop = route[1];
      int sendSize = static_cast<int>(buffer.size());
      MPI_Send(&sendSize, 1, MPI_INT, nextHop, 0, MPI_COMM_WORLD);
      if (sendSize > 0) {
        MPI_Send(received.data(), sendSize, MPI_INT, nextHop, 1, MPI_COMM_WORLD);
      }
    }
  } else if (onRoute) {
    int prevHop = route[myIndex - 1];
    int recvSize = 0;
    MPI_Recv(&recvSize, 1, MPI_INT, prevHop, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    received.resize(recvSize);
    if (recvSize > 0) {
      MPI_Recv(received.data(), recvSize, MPI_INT, prevHop, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (currentRank_ != dst && myIndex + 1 < routeLen) {
      int nextHop = route[myIndex + 1];
      MPI_Send(&recvSize, 1, MPI_INT, nextHop, 0, MPI_COMM_WORLD);
      if (recvSize > 0) {
        MPI_Send(received.data(), recvSize, MPI_INT, nextHop, 1, MPI_COMM_WORLD);
      }
    }
  }
}

void TorusNetworkMpi::SaveResult(int dst, const std::vector<int> &received, const std::vector<int> &route) {
  if (currentRank_ == dst) {
    localOutput_.payload = received;
    localOutput_.path = route;
    GetOutput() = localOutput_;
  } else {
    GetOutput() = OutType{};
  }
}

bool TorusNetworkMpi::PostProcessingImpl() {
  return true;
}

}  // namespace klimov_m_torus