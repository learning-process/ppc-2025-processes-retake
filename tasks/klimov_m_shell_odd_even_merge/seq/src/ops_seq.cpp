#include "klimov_m_shell_odd_even_merge/seq/include/ops_seq.hpp"

#include <vector>

namespace klimov_m_shell_odd_even_merge {

ShellBatcherSEQ::ShellBatcherSEQ(const InputType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
}

bool ShellBatcherSEQ::ValidationImpl() { return !GetInput().empty(); }
bool ShellBatcherSEQ::PreProcessingImpl() { return true; }
bool ShellBatcherSEQ::PostProcessingImpl() { return true; }

bool ShellBatcherSEQ::RunImpl() {
  std::vector<int> data = GetInput();
  const size_t n = data.size();

  for (size_t gap = n / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < n; ++i) {
      int tmp = data[i];
      size_t j = i;
      while (j >= gap && data[j - gap] > tmp) {
        data[j] = data[j - gap];
        j -= gap;
      }
      data[j] = tmp;
    }
  }

  GetOutput() = data;
  return true;
}

}  // namespace klimov_m_shell_odd_even_merge