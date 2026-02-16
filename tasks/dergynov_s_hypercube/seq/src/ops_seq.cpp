#include "dergynov_s_hypercube/seq/include/ops_seq.hpp"

#include "dergynov_s_hypercube/common/include/common.hpp"

namespace dergynov_s_hypercube {

DergynovSHypercubeSEQ::DergynovSHypercubeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool DergynovSHypercubeSEQ::ValidationImpl() {
  return true;
}

bool DergynovSHypercubeSEQ::PreProcessingImpl() {
  return true;
}

bool DergynovSHypercubeSEQ::RunImpl() {
  GetOutput() = GetInput()[2];
  return true;
}

bool DergynovSHypercubeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace dergynov_s_hypercube