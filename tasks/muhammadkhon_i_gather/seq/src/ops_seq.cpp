#include "muhammadkhon_i_gather/seq/include/ops_seq.hpp"

#include <vector>

#include "muhammadkhon_i_gather/common/include/common.hpp"

namespace muhammadkhon_i_gather {

MuhammadkhonIGatherSEQ::MuhammadkhonIGatherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool MuhammadkhonIGatherSEQ::ValidationImpl() {
  int root = GetInput().root;
  return root == 0;
}

bool MuhammadkhonIGatherSEQ::PreProcessingImpl() {
  return true;
}

bool MuhammadkhonIGatherSEQ::RunImpl() {
  const std::vector<double> &send_data = GetInput().send_data;
  GetOutput().recv_data = send_data;
  return true;
}

bool MuhammadkhonIGatherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace muhammadkhon_i_gather
