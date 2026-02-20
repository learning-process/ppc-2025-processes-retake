#include "shkryleva_s_vec_min_val/seq/include/ops_seq.hpp"

#include <cstddef>
#include <limits>
#include <vector>

#include "shkryleva_s_vec_min_val/common/include/common.hpp"

namespace shkryleva_s_vec_min_val {

ShkrylevaSVecMinValSEQ::ShkrylevaSVecMinValSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShkrylevaSVecMinValSEQ::ValidationImpl() {
  return true;
}

bool ShkrylevaSVecMinValSEQ::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::max();
  return true;
}

bool ShkrylevaSVecMinValSEQ::RunImpl() {
  const auto &input = GetInput();

  if (input.empty()) {
    GetOutput() = std::numeric_limits<int>::max();
    return true;
  }

  int min_val = input.front();
  for (std::size_t i = 1; i < input.size(); ++i) {
    if (input[i] < min_val) {
      min_val = input[i];
    }
  }

  GetOutput() = min_val;
  return true;
}

bool ShkrylevaSVecMinValSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shkryleva_s_vec_min_val
