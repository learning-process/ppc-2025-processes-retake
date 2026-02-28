#pragma once

#include <utility>
#include <vector>

#include "klimov_m_torus/common/include/common.hpp"
#include "task/include/task.hpp"

namespace klimov_m_torus {

class TorusNetworkMpi : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit TorusNetworkMpi(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::pair<int, int> CalculateGridDimensions(int totalProcs);
  static int RankFromCoordinates(int row, int col, int rows, int cols);
  static std::pair<int, int> CoordinatesFromRank(int rank, int cols);
  static std::vector<int> BuildRoute(int rows, int cols, int from, int to);

  void BroadcastSourceAndDestination(int &src, int &dst);
  void BroadcastDataSize(int src, int &dataSize) const;
  std::vector<int> PrepareDataBuffer(int src, int dataSize) const;
  void ForwardData(int src, int dst, const std::vector<int> &route,
                   const std::vector<int> &buffer, std::vector<int> &received) const;
  void SaveResult(int dst, const std::vector<int> &received, const std::vector<int> &route);

  InType localInput_{};
  OutType localOutput_{};

  int currentRank_{0};
  int worldSize_{0};

  int gridRows_{1};
  int gridCols_{1};
};

}  // namespace klimov_m_torus