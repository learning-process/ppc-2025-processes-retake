#pragma once

#include "fedoseev_gaussian_method_horizontal_strip_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace fedoseev_gaussian_method_horizontal_strip_scheme {

class FedoseevTestTaskSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit FedoseevTestTaskSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace fedoseev_gaussian_method_horizontal_strip_scheme
