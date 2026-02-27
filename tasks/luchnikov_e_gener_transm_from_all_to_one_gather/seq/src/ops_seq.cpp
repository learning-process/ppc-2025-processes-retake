#include "luchnikov_e_gener_transm_from_all_to_one_gather/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

LuchnikovEGenerTransmFromAllToOneGatherSEQ::LuchnikovEGenerTransmFromAllToOneGatherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool LuchnikovEGenerTransmFromAllToOneGatherSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool LuchnikovEGenerTransmFromAllToOneGatherSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool LuchnikovEGenerTransmFromAllToOneGatherSEQ::RunImpl() {
  return true;
}

bool LuchnikovEGenerTransmFromAllToOneGatherSEQ::PostProcessingImpl() {
  std::sort(GetOutput().begin(), GetOutput().end());
  return true;
}

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
