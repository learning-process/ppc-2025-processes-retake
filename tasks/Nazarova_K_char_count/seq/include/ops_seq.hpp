#pragma once

#include "Nazarova_K_char_count/common/include/common.hpp"

namespace nazarova_k_char_count_processes {

class NazarovaKCharCountSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit NazarovaKCharCountSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nazarova_k_char_count_processes
