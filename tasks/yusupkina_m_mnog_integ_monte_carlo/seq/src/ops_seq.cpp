#include "yusupkina_m_mnog_integ_monte_carlo/seq/include/ops_seq.hpp"

#include <cstdint>
#include <random>

#include "yusupkina_m_mnog_integ_monte_carlo/common/include/common.hpp"

namespace yusupkina_m_mnog_integ_monte_carlo {

YusupkinaMMnogIntegMonteCarloSEQ::YusupkinaMMnogIntegMonteCarloSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool YusupkinaMMnogIntegMonteCarloSEQ::ValidationImpl() {
  const auto &input = GetInput();
  return input.num_points > 0 && input.x_min <= input.x_max && input.y_min <= input.y_max;
}

bool YusupkinaMMnogIntegMonteCarloSEQ::PreProcessingImpl() {
  return true;
}

bool YusupkinaMMnogIntegMonteCarloSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &result = GetOutput();

  if (input.num_points <= 0) {
    result = 0.0;
    return true;
  }

  const double area = (input.x_max - input.x_min) * (input.y_max - input.y_min);
  if (area <= 0.0) {
    result = 0.0;
    return true;
  }

  const double x_min = input.x_min;
  const double x_max = input.x_max;
  const double y_min = input.y_min;
  const double y_max = input.y_max;
  const auto &func = input.f;
  const int64_t n = input.num_points;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist_x(x_min, x_max);
  std::uniform_real_distribution<double> dist_y(y_min, y_max);

  double sum = 0.0;

  for (int64_t i = 0; i < n; i++) {
    double x = dist_x(gen);
    double y = dist_y(gen);
    sum += func(x, y);
  }

  result = (sum / static_cast<double>(n)) * area;

  return true;
}

bool YusupkinaMMnogIntegMonteCarloSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace yusupkina_m_mnog_integ_monte_carlo
