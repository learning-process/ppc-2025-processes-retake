#pragma once

#include <vector>

#include "sokolov_k_shell_simple_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sokolov_k_shell_simple_merge {

class SokolovKShellSimpleMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SokolovKShellSimpleMergeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> data_;
};

}  // namespace sokolov_k_shell_simple_merge
