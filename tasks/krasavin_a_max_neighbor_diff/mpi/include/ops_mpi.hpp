#pragma once

#include "krasavin_a_max_neighbor_diff/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krasavin_a_max_neighbor_diff {

class KrasavinAMaxNeighborDiffMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KrasavinAMaxNeighborDiffMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace krasavin_a_max_neighbor_diff