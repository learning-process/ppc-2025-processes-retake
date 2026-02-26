#include "fedoseev_multi_step_scheme_parallelization_by_characteristics/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>

#include "fedoseev_multi_step_scheme_parallelization_by_characteristics/common/include/common.hpp"

namespace fedoseev_multi_step_scheme_parallelization_by_characteristics {

FedoseevMultiStepSchemeMPI::FedoseevMultiStepSchemeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool FedoseevMultiStepSchemeMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0.0);
}

bool FedoseevMultiStepSchemeMPI::PreProcessingImpl() {
  GetOutput() = 1.0;
  return true;
}

bool FedoseevMultiStepSchemeMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  InType input = GetInput();
  double local_result = 0.0;
  double global_result = 0.0;

  int points_per_process = input / size;
  int start_x = rank * points_per_process;
  int end_x = (rank == size - 1) ? input : (rank + 1) * points_per_process;

  for (int x = start_x; x < end_x; ++x) {
    for (int y = 0; y < input; ++y) {
      double val = std::sin(x * 0.1) * std::cos(y * 0.1);
      local_result += val;

      for (int step = 0; step < 3; ++step) {
        double delta = 0.01;
        val = std::sin((x + step * delta) * 0.1) * std::cos((y + step * delta) * 0.1);
        local_result = std::max(local_result, val);
      }
    }
  }

  MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_result / (input * input);
  }

  MPI_Bcast(&GetOutput(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool FedoseevMultiStepSchemeMPI::PostProcessingImpl() {
  GetOutput() = std::abs(GetOutput());
  return true;
}

}  // namespace fedoseev_multi_step_scheme_parallelization_by_characteristics
