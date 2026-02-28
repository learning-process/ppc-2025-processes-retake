#include "savva_d_zeidel_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "savva_d_zeidel_method/common/include/common.hpp"

namespace savva_d_zeidel_method {

SavvaDZeidelMPI::SavvaDZeidelMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>{};
}

bool SavvaDZeidelMPI::ValidationImpl() {
  const auto &in = GetInput();

  if (in.n < 0 || in.a.size() != static_cast<size_t>(in.n) * static_cast<size_t>(in.n) ||
      in.b.size() != static_cast<size_t>(in.n)) {
    return false;
  }

  for (int i = 0; i < in.n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < in.n; ++j) {
      if (i != j) {
        sum += std::abs(in.a[(i * in.n) + j]);
      }
    }
    if (std::abs(in.a[(i * in.n) + i]) <= sum) {
      return false;
    }
  }

  return true;
}

bool SavvaDZeidelMPI::PreProcessingImpl() {
  GetOutput().assign(GetInput().n, 0.0);
  return true;
}

void SavvaDZeidelMPI::RunSeidelIterations(int n, int local_rows, int local_offset, const double *local_data_a,
                                          const double *local_data_b, std::vector<double> &x, const int *counts2,
                                          const int *displacements2) {
  for (int iter = 0; iter < 1000; ++iter) {
    double local_max_error = 0.0;

    for (int i = 0; i < local_rows; ++i) {
      int index = local_offset + i;
      double result = local_data_b[i];

      for (int j = 0; j < n; ++j) {
        if (j != index) {
          result -= local_data_a[(static_cast<size_t>(i) * n) + j] * x[j];
        }
      }

      double diag_element = local_data_a[(static_cast<size_t>(i) * n) + index];
      double x_actual = result / diag_element;

      double current_diff = std::abs(x_actual - x[index]);
      local_max_error = std::max(local_max_error, current_diff);
      x[index] = x_actual;
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), counts2, displacements2, MPI_DOUBLE, MPI_COMM_WORLD);

    double global_max_error = 0.0;
    MPI_Allreduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (global_max_error < 0.00001) {
      break;
    }
  }
}

bool SavvaDZeidelMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int local_offset = 0;
  int local_rows = 0;
  double *local_data_a = nullptr;
  double *local_data_b = nullptr;
  double *global_data_a = nullptr;
  double *global_data_b = nullptr;
  int *counts = new int[size]();
  int *displacements = new int[size]();
  int *counts2 = new int[size]();
  int *displacements2 = new int[size]();

  int n = 0;

  if (rank == 0) {
    n = GetInput().n;
    global_data_a = GetInput().a.data();
    global_data_b = GetInput().b.data();
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0) {
    delete[] counts;
    delete[] displacements;
    delete[] counts2;
    delete[] displacements2;
    auto &x = GetOutput();
    x = std::vector<double>{};
    return true;
  }

  int elements_per_proc = n / size;
  int remainder = n % size;
  int offset = 0;
  int offset2 = 0;

  for (int i = 0; i < size; ++i) {
    counts[i] = (elements_per_proc + (i < remainder ? 1 : 0)) * n;
    displacements[i] = offset;
    if (i == rank) {
      local_offset = offset2;
    }
    offset += counts[i];
    counts2[i] = (elements_per_proc + (i < remainder ? 1 : 0));
    displacements2[i] = offset2;
    offset2 += counts2[i];
  }
  local_rows = (elements_per_proc + (rank < remainder ? 1 : 0));
  const std::size_t local_size_a = static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(n);
  local_data_a = new double[local_size_a];
  local_data_b = new double[local_rows];

  MPI_Scatterv(global_data_a, counts, displacements, MPI_DOUBLE, local_data_a, local_rows * n, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
  MPI_Scatterv(global_data_b, counts2, displacements2, MPI_DOUBLE, local_data_b, local_rows, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  auto &x = GetOutput();
  x.assign(n, 0.0);

  RunSeidelIterations(n, local_rows, local_offset, local_data_a, local_data_b, x, counts2, displacements2);

  delete[] counts;
  delete[] displacements;
  delete[] counts2;
  delete[] displacements2;
  delete[] local_data_a;
  delete[] local_data_b;

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool SavvaDZeidelMPI::PostProcessingImpl() {
  return true;
}

}  // namespace savva_d_zeidel_method
