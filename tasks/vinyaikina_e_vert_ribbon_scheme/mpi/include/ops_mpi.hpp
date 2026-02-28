#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "vinyaikina_e_vert_ribbon_scheme/common/include/common.hpp"

namespace vinyaikina_e_vert_ribbon_scheme {

class VinyaikinaEVertRibbonSchemeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VinyaikinaEVertRibbonSchemeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int rows_ = 0;
  int cols_ = 0;
  std::vector<int> matrix_;
  std::vector<int> vector_;
  std::vector<int> result_;
};

}  // namespace vinyaikina_e_vert_ribbon_scheme
