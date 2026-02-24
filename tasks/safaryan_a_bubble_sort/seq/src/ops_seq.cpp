#include "safaryan_a_bubble_sort/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "safaryan_a_bubble_sort/common/include/common.hpp"

namespace safaryan_a_bubble_sort {

SafaryanABubbleSortSEQ::SafaryanABubbleSortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SafaryanABubbleSortSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool SafaryanABubbleSortSEQ::PreProcessingImpl() {
  return true;
}

bool SafaryanABubbleSortSEQ::RunImpl() {
  std::vector<int> data = GetInput();
  size_t n = data.size();

  int temp = 0;

  std::vector<int> out;
  out.resize(n);
  for (size_t i = 0; i < n; out[n - i - 1] = data[n - i - 1], i++) {
    for (size_t j = 0; j < n - i - 1; j++) {
      if (data[j] > data[j + 1]) {
        temp = data[j];
        data[j] = data[j + 1];
        data[j + 1] = temp;
      }
    }
  }
  GetOutput() = out;
  return true;
}

bool SafaryanABubbleSortSEQ::PostProcessingImpl() {
  return true;
}
}  // namespace safaryan_a_bubble_sort