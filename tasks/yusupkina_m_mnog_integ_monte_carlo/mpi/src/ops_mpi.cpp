#include "yusupkina_m_mnog_integ_monte_carlo/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdint>
#include <random>

#include "yusupkina_m_mnog_integ_monte_carlo/common/include/common.hpp"

namespace yusupkina_m_mnog_integ_monte_carlo {

YusupkinaMMnogIntegMonteCarloMPI::YusupkinaMMnogIntegMonteCarloMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool YusupkinaMMnogIntegMonteCarloMPI::ValidationImpl() {
  const auto &input = GetInput();
  return input.num_points > 0 && input.x_min <= input.x_max && input.y_min <= input.y_max;
}

bool YusupkinaMMnogIntegMonteCarloMPI::PreProcessingImpl() {
  return true;
}

bool YusupkinaMMnogIntegMonteCarloMPI::RunImpl() {
  const auto &input = GetInput();
  auto &result = GetOutput();

  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int64_t points = input.num_points;
  const double x_min = input.x_min;
  const double x_max = input.x_max;
  const double y_min = input.y_min;
  const double y_max = input.y_max;

  MPI_Bcast(&points, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
  if (points <= 0) {
    result = 0.0;
    return true;
  }

  double area = (x_max - x_min) * (y_max - y_min);
  MPI_Bcast(&area, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (area <= 0.0) {
    result = 0.0;
    return true;
  }

  int64_t base_points = points / size;
  int64_t remainder = points % size;
  int64_t local_points = base_points + (rank < remainder ? 1 : 0);

  std::random_device rd;
  std::mt19937 gen(rd() + rank);
  std::uniform_real_distribution<double> dist_x(x_min, x_max);
  std::uniform_real_distribution<double> dist_y(y_min, y_max);

  double local_sum = 0.0;
  for (int64_t i = 0; i < local_points; i++) {
    double x = dist_x(gen);
    double y = dist_y(gen);
    local_sum += input.f(x, y);
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    result = (global_sum / static_cast<double>(points)) * area;
  }
  MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool YusupkinaMMnogIntegMonteCarloMPI::PostProcessingImpl() {
  return true;
}

}  // namespace yusupkina_m_mnog_integ_monte_carlo
