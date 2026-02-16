#pragma once

#include <vector>

#include "dergynov_s_hypercube/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dergynov_s_hypercube {

class DergynovSHypercubeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit DergynovSHypercubeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int CalcDim(int size);
  static std::vector<int> BuildPath(int src, int dst, int dim);
  static void FindPos(int rank, const std::vector<int> &path, int &pos, int &next, int &prev);

  static void SendVec(const std::vector<int> &data, int to);
  static void RecvVec(std::vector<int> &data, int from);
  static void BusyWork(int iters);
};

}  // namespace dergynov_s_hypercube