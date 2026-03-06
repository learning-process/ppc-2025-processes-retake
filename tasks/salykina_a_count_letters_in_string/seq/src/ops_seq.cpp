#include "salykina_a_count_letters_in_string/seq/include/ops_seq.hpp"

#include <cctype>
#include <string>

#include "salykina_a_count_letters_in_string/common/include/common.hpp"

namespace salykina_a_count_letters_in_string {

SalykinaACountLettersSEQ::SalykinaACountLettersSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SalykinaACountLettersSEQ::ValidationImpl() {
  return !GetInput().empty() && (GetOutput() == 0);
}

bool SalykinaACountLettersSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SalykinaACountLettersSEQ::RunImpl() {
  const std::string &input = GetInput();
  int count = 0;
  for (char c : input) {
    if (std::isalpha(static_cast<unsigned char>(c)) != 0) {
      count++;
    }
  }
  GetOutput() = count;
  return GetOutput() >= 0;
}

bool SalykinaACountLettersSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace salykina_a_count_letters_in_string
