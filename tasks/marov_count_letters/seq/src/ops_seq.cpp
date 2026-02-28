#include "marov_count_letters/seq/include/ops_seq.hpp"

#include <cctype>
#include <string>

#include "marov_count_letters/common/include/common.hpp"

namespace marov_count_letters {

MarovCountLettersSeq::MarovCountLettersSeq(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MarovCountLettersSeq::ValidationImpl() {
  return true;
}

bool MarovCountLettersSeq::PreProcessingImpl() {
  return true;
}

bool MarovCountLettersSeq::RunImpl() {
  const std::string& input_str = GetInput();
  int count = 0;

  for (char c : input_str) {
    if (std::isalpha(static_cast<unsigned char>(c)) != 0) {
      count++;
    }
  }

  GetOutput() = count;
  return true;
}

bool MarovCountLettersSeq::PostProcessingImpl() {
  return true;
}

}  // namespace marov_count_letters
