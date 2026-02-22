#include "tsarkov_k_lexicographic_string_compare/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>

#include "tsarkov_k_lexicographic_string_compare/common/include/common.hpp"

namespace tsarkov_k_lexicographic_string_compare {

TsarkovKLexicographicStringCompareSEQ::TsarkovKLexicographicStringCompareSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TsarkovKLexicographicStringCompareSEQ::ValidationImpl() {
  return GetOutput() == 0;
}

bool TsarkovKLexicographicStringCompareSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool TsarkovKLexicographicStringCompareSEQ::RunImpl() {
  const auto &[first_str, second_str] = GetInput();

  const std::size_t min_length = std::min(first_str.size(), second_str.size());
  for (std::size_t index = 0; index < min_length; ++index) {
    const auto first_ch = static_cast<unsigned char>(first_str[index]);
    const auto second_ch = static_cast<unsigned char>(second_str[index]);

    if (first_ch < second_ch) {
      GetOutput() = 1;
      return true;
    }
    if (first_ch > second_ch) {
      GetOutput() = 0;
      return true;
    }
  }

  GetOutput() = (first_str.size() <= second_str.size()) ? 1 : 0;
  return true;
}

bool TsarkovKLexicographicStringCompareSEQ::PostProcessingImpl() {
  return (GetOutput() == 0) || (GetOutput() == 1);
}

}  // namespace tsarkov_k_lexicographic_string_compare
