#include "zyuzin_n_sum_elements_of_matrix/seq/include/ops_seq.hpp"

#include <cstddef>
#include <numeric>

#include "zyuzin_n_sum_elements_of_matrix/common/include/common.hpp"

namespace zyuzin_n_sum_elements_of_matrix {

ZyuzinNSumElementsOfMatrixSEQ::ZyuzinNSumElementsOfMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool ZyuzinNSumElementsOfMatrixSEQ::ValidationImpl() {
  const auto &matrix = GetInput();
  return std::get<0>(matrix) > 0 && std::get<1>(matrix) > 0 &&
         std::get<2>(matrix).size() ==
             static_cast<std::size_t>(std::get<0>(matrix)) * static_cast<std::size_t>(std::get<1>(matrix));
}

bool ZyuzinNSumElementsOfMatrixSEQ::PreProcessingImpl() {
  return true;
}

bool ZyuzinNSumElementsOfMatrixSEQ::RunImpl() {
  const auto &matrix = GetInput();
  GetOutput() = std::accumulate(std::get<2>(matrix).begin(), std::get<2>(matrix).end(), 0.0);
  return true;
}

bool ZyuzinNSumElementsOfMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zyuzin_n_sum_elements_of_matrix
