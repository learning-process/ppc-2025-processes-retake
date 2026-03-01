#include "denisov_a_ring/seq/include/ops_seq.hpp"

#include "denisov_a_ring/common/include/common.hpp"

namespace denisov_a_ring {

RingTopologySEQ::RingTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool RingTopologySEQ::ValidationImpl() {
  const auto &in = GetInput();
  return (in.source >= 0) && (in.destination >= 0);
}

bool RingTopologySEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool RingTopologySEQ::RunImpl() {
  const auto &in = GetInput();
  GetOutput() = in.data;
  return true;
}

bool RingTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace denisov_a_ring
