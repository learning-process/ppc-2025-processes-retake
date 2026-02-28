#pragma once
#include <vector>
#include "solonin_v_scatter/common/include/common.hpp"
#include "task/include/task.hpp"

namespace solonin_v_scatter {

// Custom implementation of MPI_Scatter without using MPI_Scatter itself.
// Root sends send_count elements to each process (including itself).
class SoloninVScatterMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() { return ppc::task::TypeOfTask::kMPI; }
  explicit SoloninVScatterMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int rank_{0};
  int world_size_{1};
};

}  // namespace solonin_v_scatter
