#pragma once

#include "savva_d_zeidel_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace savva_d_zeidel_method {

class SavvaDZeidelSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SavvaDZeidelSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace savva_d_zeidel_method
