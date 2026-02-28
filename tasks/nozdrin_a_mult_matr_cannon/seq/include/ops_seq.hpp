#pragma once

#include "nozdrin_a_mult_matr_cannon/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nozdrin_a_mult_matr_cannon {

class NozdrinAMultMatrCannonSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit NozdrinAMultMatrCannonSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nozdrin_a_mult_matr_cannon
