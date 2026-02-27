#pragma once

#include <vector>

#include "Nazarova_K_Conv_hull_const_Grah/common/include/common.hpp"

namespace nazarova_k_conv_hull_const_grah_processes {

class NazarovaKConvHullConstGrahSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit NazarovaKConvHullConstGrahSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<Point> points_;
};

}  // namespace nazarova_k_conv_hull_const_grah_processes
