#pragma once

#include <array>
#include <vector>

#include "nazyrov_a_global_opt_2d/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nazyrov_a_global_opt_2d {

class GlobalOpt2dMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GlobalOpt2dMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  struct TrialPoint {
    double x;
    double y;
    double z;
  };

  int world_rank_{0};
  int world_size_{1};

  std::vector<double> t_points_;
  std::vector<TrialPoint> trials_;
  double lip_est_{1.0};
  static constexpr int kPeanoLevel = 10;

  static void PeanoMap(double t_val, int level, double &ux, double &uy);
  double ToX(double t_val);
  double ToY(double t_val);
  void SortTrials();
  double ComputeLipschitz();

  void InitTrials();
  void DistributeAndCompute(double m_val, std::vector<double> &all_chars);
  std::array<double, 3> SelectBestAndAddTrial(const std::vector<double> &all_chars, double m_val, int &converge_flag);
  void FindBestResult();
};

}  // namespace nazyrov_a_global_opt_2d
