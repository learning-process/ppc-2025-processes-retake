#include "tsarkov_k_monte_carlo_integration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

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

[[nodiscard]] int ComputeLocalSamples(const int total_samples, const int world_rank, const int world_size) {
  const int base = total_samples / world_size;
  const int rem = total_samples % world_size;
  return base + ((world_rank < rem) ? 1 : 0);
}

[[nodiscard]] double LocalMonteCarloSum(const int dimension, const int local_samples, const std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<double> point(static_cast<std::size_t>(dimension), 0.0);

  double local_sum = 0.0;
  for (int sample_index = 0; sample_index < local_samples; ++sample_index) {
    for (int dim_index = 0; dim_index < dimension; ++dim_index) {
      point[static_cast<std::size_t>(dim_index)] = dist(rng);
    }
    local_sum += IntegrandExpMinusSquaredNorm(point);
  }

  return local_sum;
}

}  // namespace

TsarkovKMonteCarloIntegrationMPI::TsarkovKMonteCarloIntegrationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TsarkovKMonteCarloIntegrationMPI::ValidationImpl() {
  if (!HasValidInputShape(GetInput())) {
    return false;
  }
  const int dimension = GetDimension(GetInput());
  const int samples = GetSamples(GetInput());
  return (dimension > 0) && (samples > 0);
}

bool TsarkovKMonteCarloIntegrationMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool TsarkovKMonteCarloIntegrationMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int dimension = GetDimension(GetInput());
  const int total_samples = GetSamples(GetInput());
  const std::uint32_t base_seed = GetSeed(GetInput());

  if ((dimension <= 0) || (total_samples <= 0) || (world_size <= 0)) {
    return false;
  }

  const int local_samples = ComputeLocalSamples(total_samples, world_rank, world_size);
  const std::uint32_t rank_seed = base_seed + static_cast<std::uint32_t>(world_rank * 1000003);

  const double local_sum = LocalMonteCarloSum(dimension, local_samples, rank_seed);

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    GetOutput() = global_sum / static_cast<double>(total_samples);
  }

  MPI_Bcast(&GetOutput(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return std::isfinite(GetOutput());
}

bool TsarkovKMonteCarloIntegrationMPI::PostProcessingImpl() {
  return std::isfinite(GetOutput());
}

}  // namespace tsarkov_k_monte_carlo_integration
