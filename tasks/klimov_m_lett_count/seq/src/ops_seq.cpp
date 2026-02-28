#include "klimov_m_lett_count/seq/include/ops_seq.hpp"

#include <cctype>

namespace klimov_m_lett_count {

KlimovMLettCountSEQ::KlimovMLettCountSEQ(const InputType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

int KlimovMLettCountSEQ::CountLettersInString(const char *data, int length) {
  int count = 0;
  if (length <= 0) {
    return 0;
  }
  for (int i = 0; i < length; ++i) {
    if (std::isalpha(static_cast<unsigned char>(data[i]))) {
      ++count;
    }
  }
  return count;
}

bool KlimovMLettCountSEQ::ValidationImpl() {
  return true;
}

bool KlimovMLettCountSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return GetOutput() == 0;
}

bool KlimovMLettCountSEQ::RunImpl() {
  GetOutput() = CountLettersInString(GetInput().data(), static_cast<int>(GetInput().size()));
  return true;
}

bool KlimovMLettCountSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace klimov_m_lett_count
