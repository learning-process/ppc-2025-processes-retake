#include "salena_s_vec_min_val/seq/include/ops_seq.hpp"
#include <algorithm>
#include <limits>

namespace salena_s_vec_min_val {

TestTaskSEQ::TestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TestTaskSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool TestTaskSEQ::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::max();
  return true;
}

bool TestTaskSEQ::RunImpl() {
  for (int val : GetInput()) {
    GetOutput() = std::min(GetOutput(), val);
  }
  return true;
}

bool TestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace salena_s_vec_min_val