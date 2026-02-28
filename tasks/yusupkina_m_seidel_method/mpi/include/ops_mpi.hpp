#pragma once

#include <vector>

#include "yusupkina_m_seidel_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace yusupkina_m_seidel_method {

class YusupkinaMSeidelMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit YusupkinaMSeidelMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void RunOneIteration(int n, int local_rows, int start_row,
                              const std::vector<double>& local_A,
                              const std::vector<double>& local_b,
                              std::vector<double>& x,
                              double& local_error);
};

}  // namespace yusupkina_m_seidel_method