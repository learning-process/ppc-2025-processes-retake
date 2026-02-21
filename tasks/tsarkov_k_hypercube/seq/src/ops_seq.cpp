#include "tsarkov_k_hypercube/seq/include/ops_seq.hpp"

#include "tsarkov_k_hypercube/common/include/common.hpp"

namespace tsarkov_k_hypercube {
namespace {

[[nodiscard]] bool HasValidInputShape(const InType &input_data) {
  return input_data.size() == 3U;
}

}  // namespace

TsarkovKHypercubeSEQ::TsarkovKHypercubeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TsarkovKHypercubeSEQ::ValidationImpl() {
  const InType &input_data = GetInput();
  if (!HasValidInputShape(input_data)) {
    return false;
  }
  return input_data[2] >= 0;
}

bool TsarkovKHypercubeSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool TsarkovKHypercubeSEQ::RunImpl() {
  const InType &input_data = GetInput();
  GetOutput() = input_data[2];
  return true;
}

bool TsarkovKHypercubeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tsarkov_k_hypercube
