#pragma once

#include "yusupkina_m_seidel_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace yusupkina_m_seidel_method {

class YusupkinaMSeidelMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit YusupkinaMSeidelMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace yusupkina_m_seidel_method
