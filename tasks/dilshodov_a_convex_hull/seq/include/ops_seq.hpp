#pragma once

#include "dilshodov_a_convex_hull/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dilshodov_a_convex_hull {

class ConvexHullSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ConvexHullSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int width_{0};
  int height_{0};
};

}  // namespace dilshodov_a_convex_hull
