#include "sokolov_k_shell_simple_merge/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "sokolov_k_shell_simple_merge/common/include/common.hpp"

namespace sokolov_k_shell_simple_merge {

SokolovKShellSimpleMergeSEQ::SokolovKShellSimpleMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SokolovKShellSimpleMergeSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool SokolovKShellSimpleMergeSEQ::PreProcessingImpl() {
  int n = GetInput();
  data_.resize(n);
  for (int i = 0; i < n; i++) {
    data_[i] = n - i;
  }
  return true;
}

bool SokolovKShellSimpleMergeSEQ::RunImpl() {
  int n = static_cast<int>(data_.size());
  int gap = 1;
  while (gap < n / 3) {
    gap = (3 * gap) + 1;
  }
  while (gap >= 1) {
    for (int i = gap; i < n; i++) {
      int temp = data_[i];
      int j = i;
      while (j >= gap && data_[j - gap] > temp) {
        data_[j] = data_[j - gap];
        j -= gap;
      }
      data_[j] = temp;
    }
    gap /= 3;
  }
  return true;
}

bool SokolovKShellSimpleMergeSEQ::PostProcessingImpl() {
  for (size_t i = 1; i < data_.size(); i++) {
    if (data_[i - 1] > data_[i]) {
      return false;
    }
  }
  GetOutput() = GetInput();
  return true;
}

}  // namespace sokolov_k_shell_simple_merge
