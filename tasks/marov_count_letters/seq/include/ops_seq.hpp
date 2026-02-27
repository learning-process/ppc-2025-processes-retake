#pragma once

#include "marov_count_letters/common/include/common.hpp"
#include "task/include/task.hpp"

namespace marov_count_letters {

class MarovCountLettersSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MarovCountLettersSEQ(const InType& in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace marov_count_letters
