#include "fedoseev_multi_step_scheme_parallelization_by_characteristics/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>

#include "fedoseev_multi_step_scheme_parallelization_by_characteristics/common/include/common.hpp"

namespace fedoseev_multi_step_scheme_parallelization_by_characteristics {

FedoseevMultiStepSchemeSEQ::FedoseevMultiStepSchemeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool FedoseevMultiStepSchemeSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0.0);
}

bool FedoseevMultiStepSchemeSEQ::PreProcessingImpl() {
  GetOutput() = 1.0;
  return true;
}

bool FedoseevMultiStepSchemeSEQ::RunImpl() {
  InType input = GetInput();
  double result = 0.0;
  for (int i = 0; i < input; ++i) {
    for (int j = 0; j < input; ++j) {
      double val = std::sin(i * 0.1) * std::cos(j * 0.1);
      result += val;
      for (int step = 0; step < 3; ++step) {
        double delta = 0.01;
        val = std::sin((i + step * delta) * 0.1) * std::cos((j + step * delta) * 0.1);
        result = std::max(result, val);
      }
    }
  }

  GetOutput() = result / (input * input);

  return true;
}

bool FedoseevMultiStepSchemeSEQ::PostProcessingImpl() {
  GetOutput() = std::abs(GetOutput());
  return true;
}

}  // namespace fedoseev_multi_step_scheme_parallelization_by_characteristics
