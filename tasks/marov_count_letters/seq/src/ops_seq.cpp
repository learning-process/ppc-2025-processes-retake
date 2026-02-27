#include "marov_count_letters/seq/include/ops_seq.hpp"

#include <cctype>
#include <string>

#include "marov_count_letters/common/include/common.hpp"

namespace marov_count_letters {

MarovCountLettersSEQ::MarovCountLettersSEQ(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MarovCountLettersSEQ::ValidationImpl() {
  return true;
}

bool MarovCountLettersSEQ::PreProcessingImpl() {
  return true;
}

bool MarovCountLettersSEQ::RunImpl() {
  const std::string& input_str = GetInput();
  int count = 0;

  for (char c : input_str) {
    if (std::isalpha(static_cast<unsigned char>(c))) {
      count++;
    }
  }

  GetOutput() = count;
  return true;
}

bool MarovCountLettersSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace marov_count_letters
