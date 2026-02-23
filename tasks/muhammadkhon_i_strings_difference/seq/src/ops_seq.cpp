#include "muhammadkhon_i_strings_difference/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>

#include "muhammadkhon_i_strings_difference/common/include/common.hpp"

namespace muhammadkhon_i_strings_difference {

MuhammadkhonIStringsDifferenceSEQ::MuhammadkhonIStringsDifferenceSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MuhammadkhonIStringsDifferenceSEQ::ValidationImpl() {
  return true;
}

bool MuhammadkhonIStringsDifferenceSEQ::PreProcessingImpl() {
  return true;
}

bool MuhammadkhonIStringsDifferenceSEQ::RunImpl() {
  const auto &[s1, s2] = GetInput();

  size_t min_len = std::min(s1.size(), s2.size());
  size_t max_len = std::max(s1.size(), s2.size());

  int diff_count = 0;

  for (size_t i = 0; i < min_len; ++i) {
    if (s1[i] != s2[i]) {
      diff_count++;
    }
  }
  diff_count += static_cast<int>(max_len - min_len);

  GetOutput() = diff_count;
  return true;
}

bool MuhammadkhonIStringsDifferenceSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace muhammadkhon_i_strings_difference
