#pragma once

#include <vector>

#include "kaur_a_vert_ribbon_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kaur_a_vert_ribbon_scheme {

class KaurAVertRibbonSchemeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KaurAVertRibbonSchemeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> matrix_;
  std::vector<double> vector_;
  std::vector<double> result_;
  int rows_{0};
  int cols_{0};
};

}  // namespace kaur_a_vert_ribbon_scheme