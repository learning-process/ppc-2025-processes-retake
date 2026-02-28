#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "vinyaikina_e_quicksort_simple/common/include/common.hpp"

namespace vinyaikina_e_quicksort_simple {

class VinyaikinaEQuicksortSimpleMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VinyaikinaEQuicksortSimpleMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> data_;
};

}  // namespace vinyaikina_e_quicksort_simple
