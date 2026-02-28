#include "solonin_v_scatter/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

#include "solonin_v_scatter/common/include/common.hpp"

namespace solonin_v_scatter {

SoloninVScatterSEQ::SoloninVScatterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool SoloninVScatterSEQ::ValidationImpl() {
  const auto &buf = std::get<0>(GetInput());
  int count = std::get<1>(GetInput());
  int root = std::get<2>(GetInput());
  return !buf.empty() && count > 0 && root == 0 &&
         static_cast<int>(buf.size()) >= count;
}

bool SoloninVScatterSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool SoloninVScatterSEQ::RunImpl() {
  // SEQ: single process, root gets first send_count elements
  const auto &buf = std::get<0>(GetInput());
  int count = std::get<1>(GetInput());
  GetOutput().assign(buf.begin(), buf.begin() + count);
  return true;
}

bool SoloninVScatterSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace solonin_v_scatter
