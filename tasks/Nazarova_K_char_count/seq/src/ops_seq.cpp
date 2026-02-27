#include "Nazarova_K_char_count/seq/include/ops_seq.hpp"

#include <algorithm>

#include "Nazarova_K_char_count/common/include/common.hpp"

namespace nazarova_k_char_count_processes {

NazarovaKCharCountSEQ::NazarovaKCharCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool NazarovaKCharCountSEQ::ValidationImpl() {
  return GetOutput() == 0;
}

bool NazarovaKCharCountSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool NazarovaKCharCountSEQ::RunImpl() {
  const auto &input = GetInput();
  GetOutput() = static_cast<int>(std::count(input.text.begin(), input.text.end(), input.target));
  return true;
}

bool NazarovaKCharCountSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nazarova_k_char_count_processes
