#pragma once

#include "kulikov_d_dejkstra_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kulikov_d_dejkstra_crs {

class KulikovDDijkstraCRSMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KulikovDDijkstraCRSMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace kulikov_d_dejkstra_crs
