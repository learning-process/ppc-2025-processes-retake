#include "yusupkina_m_seidel_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "yusupkina_m_seidel_method/common/include/common.hpp"

namespace yusupkina_m_seidel_method {

YusupkinaMSeidelMethodMPI::YusupkinaMSeidelMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool YusupkinaMSeidelMethodMPI::ValidationImpl() {
  const auto &in = GetInput();
  if (in.n <= 0) 
    return false;  
  if (in.matrix.size() != static_cast<size_t>(in.n * in.n) )
    return false;
  if (in.rhs.size() != static_cast<size_t>(in.n)) 
    return false; 

  for (int i = 0; i < in.n; i++) {
    double sum = 0.0;
    for (int j = 0; j < in.n; j++) {
      if (i != j) {
        sum += std::abs(in.matrix[(i * in.n) + j]);
      }
    }
    if (std::abs(in.matrix[(i * in.n) + i]) <= sum) {
      return false;  
    }
  }
  return true;
}

bool YusupkinaMSeidelMethodMPI::PreProcessingImpl() {
  auto &out = GetOutput();
  const auto &in = GetInput();
  out.assign(in.n, 0.0);
  return true;
}


void YusupkinaMSeidelMethodMPI::RunOneIteration(int n, int local_rows, int start_row,
                                                const std::vector<double>& local_A,
                                                const std::vector<double>& local_b,
                                                std::vector<double>& x,
                                                double& local_error) {
  local_error = 0.0;
  
  for (int i = 0; i < local_rows; i++) {
    int global_i = start_row + i;
    
    double sum = local_b[i];
    for (int j = 0; j < n; j++) {
      if (j != global_i) {
        sum -= local_A[i * n + j] * x[j];
      }
    }
    
    double new_x = sum / local_A[i * n + global_i];
    double error = std::abs(new_x - x[global_i]);
    if (error > local_error) 
      local_error = error;
    
    x[global_i] = new_x;
  }
}

bool YusupkinaMSeidelMethodMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const double* global_matrix_ptr = nullptr;
  const double* global_rhs_ptr = nullptr;
  int n = 0;
  
  if (rank == 0) {
    const auto &in = GetInput();
    n = in.n;
    global_matrix_ptr = in.matrix.data();
    global_rhs_ptr = in.rhs.data();
  }
  
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  auto &x = GetOutput();
  
  if (n == 0) {
    x.clear();
    return true;
  }
  
  std::vector<int> sendcounts(size);
  std::vector<int> displs(size); 
  int base = n / size;
  int remainder = n % size;
  int offset = 0;

  for (int i = 0; i < size; i++) {
    sendcounts[i] = base + (i < remainder ? 1 : 0);
    displs[i] = offset;
    offset += sendcounts[i];
  }
  
  int local_rows = sendcounts[rank];
  int start_row = displs[rank];
  
  std::vector<double> local_A(local_rows * n);
  std::vector<double> local_b(local_rows);
  
  std::vector<int> sendcounts_elem(size);
  std::vector<int> displs_elem(size);
  for (int i = 0; i < size; ++i) {
    sendcounts_elem[i] = sendcounts[i] * n;
    displs_elem[i] = displs[i] * n;
  }
  
  MPI_Scatterv(global_matrix_ptr, sendcounts_elem.data(), displs_elem.data(), MPI_DOUBLE,
               local_A.data(), local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  MPI_Scatterv(global_rhs_ptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
               local_b.data(), local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  const int max_iter = 10000;
  const double eps = 1e-6;
  
  x.assign(n, 0.0);
  
  for (int iter = 0; iter < max_iter; iter++) {
    double local_error = 0.0;
    RunOneIteration(n, local_rows, start_row, local_A, local_b, x, local_error);
    
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   x.data(), sendcounts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    double global_error = 0.0;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    if (global_error < eps) {
      break;
    }
  }
  
  return true;
}

bool YusupkinaMSeidelMethodMPI::PostProcessingImpl() {
  return true;
}


}  // namespace yusupkina_m_seidel_method