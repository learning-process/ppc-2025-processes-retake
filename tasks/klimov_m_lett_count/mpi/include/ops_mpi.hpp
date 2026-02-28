#pragma once

#include "klimov_m_lett_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace klimov_m_lett_count {

class KlimovMLettCountMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit KlimovMLettCountMPI(const InputType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static int CountLettersInSegment(const char *data, int length);

  int numProcs_ = 0;
};

}  // namespace klimov_m_lett_count