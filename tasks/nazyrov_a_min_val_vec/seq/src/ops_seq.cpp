#include "nazyrov_a_min_val_vec/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "nazyrov_a_min_val_vec/common/include/common.hpp"

namespace nazyrov_a_min_val_vec {

MinValVecSEQ::MinValVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MinValVecSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool MinValVecSEQ::PreProcessingImpl() {
  return true;
}

bool MinValVecSEQ::RunImpl() {
  const auto &input = GetInput();
  int min_elem = input[0];
  for (std::size_t i = 1; i < input.size(); ++i) {
    min_elem = std::min(input[i], min_elem);
  }
  GetOutput() = min_elem;
  return true;
}

bool MinValVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nazyrov_a_min_val_vec
