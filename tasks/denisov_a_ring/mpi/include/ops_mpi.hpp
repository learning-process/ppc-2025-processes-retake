#pragma once

#include <vector>

#include "denisov_a_ring/common/include/common.hpp"
#include "task/include/task.hpp"

namespace denisov_a_ring {

class RingTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit RingTopologyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void static SendVector(const std::vector<int> &data, int to_rank);
  void static ReceiveVector(std::vector<int> &data, int from_rank);
  void static BroadcastResult(std::vector<int> &output, int rank, int root);
};

}  // namespace denisov_a_ring
