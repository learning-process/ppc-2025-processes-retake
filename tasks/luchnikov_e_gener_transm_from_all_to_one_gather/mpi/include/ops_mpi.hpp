#pragma once

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

class LuchnikovEGenerTransmFromAllToOneGatherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LuchnikovEGenerTransmFromAllToOneGatherMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::vector<int> local_data_;
  int rank_;
  int size_;
};

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
