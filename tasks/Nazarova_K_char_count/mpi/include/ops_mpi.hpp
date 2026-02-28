#pragma once

#include "Nazarova_K_char_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nazarova_k_char_count_processes {

class NazarovaKCharCountMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NazarovaKCharCountMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nazarova_k_char_count_processes
