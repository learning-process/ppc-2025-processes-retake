#include "tsarkov_k_monte_carlo_integration/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "tsarkov_k_monte_carlo_integration/common/include/common.hpp"

namespace tsarkov_k_monte_carlo_integration {
namespace {

[[nodiscard]] bool HasValidInputShape(const InType &input_data) {
  return input_data.size() == 3U;
}

[[nodiscard]] int GetDimension(const InType &input_data) {
  return input_data[0];
}

[[nodiscard]] int GetSamples(const InType &input_data) {
  return input_data[1];
}

[[nodiscard]] std::uint32_t GetSeed(const InType &input_data) {
  return static_cast<std::uint32_t>(input_data[2]);
}

[[nodiscard]] double IntegrandExpMinusSquaredNorm(const std::vector<double> &point) {
  double sum_sq = 0.0;
  for (const double coord : point) {
    sum_sq += coord * coord;
  }
  return std::exp(-sum_sq);
}

[[nodiscard]] double MonteCarloEstimate(const int dimension, const int samples, const std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<double> point(static_cast<std::size_t>(dimension), 0.0);

  double sum_values = 0.0;
  for (int sample_index = 0; sample_index < samples; ++sample_index) {
    for (int dim_index = 0; dim_index < dimension; ++dim_index) {
      point[static_cast<std::size_t>(dim_index)] = dist(rng);
    }
    sum_values += IntegrandExpMinusSquaredNorm(point);
  }

  return sum_values / static_cast<double>(samples);
}

}  // namespace

TsarkovKMonteCarloIntegrationSEQ::TsarkovKMonteCarloIntegrationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TsarkovKMonteCarloIntegrationSEQ::ValidationImpl() {
  if (!HasValidInputShape(GetInput())) {
    return false;
  }
  const int dimension = GetDimension(GetInput());
  const int samples = GetSamples(GetInput());
  return (dimension > 0) && (samples > 0);
}

bool TsarkovKMonteCarloIntegrationSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool TsarkovKMonteCarloIntegrationSEQ::RunImpl() {
  const int dimension = GetDimension(GetInput());
  const int samples = GetSamples(GetInput());
  const std::uint32_t seed = GetSeed(GetInput());

  if ((dimension <= 0) || (samples <= 0)) {
    return false;
  }

  GetOutput() = MonteCarloEstimate(dimension, samples, seed);
  return std::isfinite(GetOutput());
}

bool TsarkovKMonteCarloIntegrationSEQ::PostProcessingImpl() {
  return std::isfinite(GetOutput());
}

}  // namespace tsarkov_k_monte_carlo_integration
