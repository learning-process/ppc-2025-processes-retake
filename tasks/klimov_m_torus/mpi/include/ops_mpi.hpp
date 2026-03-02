#pragma once

#include <utility>
#include <vector>

#include "klimov_m_torus/common/include/common.hpp"
#include "task/include/task.hpp"

namespace klimov_m_torus {

class TorusMeshCommunicator : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit TorusMeshCommunicator(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::pair<int, int> CalculateGridSize(int total_processes);
  static int CombineCoordinates(int row, int col, int rows, int cols);
  static std::pair<int, int> SplitRank(int rank, int cols);
  static std::vector<int> BuildMessageRoute(int rows, int cols, int from, int to);

  void DistributeSenderReceiver(int &src, int &dst);
  void DistributeDataLength(int src, int &len) const;
  [[nodiscard]] std::vector<int> AssembleSendBuffer(int src, int len) const;
  void RelayMessage(int src, int dst, const std::vector<int> &route, const std::vector<int> &buffer,
                    std::vector<int> &output) const;
  void SaveFinalResult(int dst, const std::vector<int> &output, const std::vector<int> &route);

  InType local_request_{};
  OutType local_response_{};

  int current_rank_{0};
  int total_ranks_{0};

  int grid_rows_{1};
  int grid_cols_{1};
};

}  // namespace klimov_m_torus
