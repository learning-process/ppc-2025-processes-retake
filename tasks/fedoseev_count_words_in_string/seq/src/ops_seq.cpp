#include "fedoseev_count_words_in_string/seq/include/ops_seq.hpp"

#include <cctype>
#include <string>

#include "fedoseev_count_words_in_string/common/include/common.hpp"

namespace fedoseev_count_words_in_string {

FedoseevCountWordsInStringSEQ::FedoseevCountWordsInStringSEQ(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FedoseevCountWordsInStringSEQ::ValidationImpl() {
  return GetOutput() == 0;
}

bool FedoseevCountWordsInStringSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool FedoseevCountWordsInStringSEQ::RunImpl() {
  const std::string &s = GetInput();
  if (s.empty()) {
    GetOutput() = 0;
    return true;
  }

  int count = 0;
  bool in_word = false;
  for (char ch : s) {
    auto uc = static_cast<unsigned char>(ch);
    bool is_space = std::isspace(uc) != 0;
    if (!is_space) {
      if (!in_word) {
        ++count;
        in_word = true;
      }
    } else {
      in_word = false;
    }
  }

  GetOutput() = count;
  return true;
}

bool FedoseevCountWordsInStringSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace fedoseev_count_words_in_string