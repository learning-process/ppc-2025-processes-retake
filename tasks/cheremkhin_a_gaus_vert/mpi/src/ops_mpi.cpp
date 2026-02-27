#include "cheremkhin_a_gaus_vert/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "cheremkhin_a_gaus_vert/common/include/common.hpp"

namespace cheremkhin_a_gaus_vert {

namespace {

constexpr double kEps = 1e-12;

struct ColRange {
  int start = 0;
  int end = 0;  // exclusive
};

inline double At(const std::vector<double> &a, int n, int r, int c) {
  return a[(static_cast<std::size_t>(r) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(c)];
}

inline std::size_t LocalAt(int row, int lc, int local_cols_count) {
  return (static_cast<std::size_t>(row) * static_cast<std::size_t>(local_cols_count)) + static_cast<std::size_t>(lc);
}

ColRange GetColRange(int rank, int size, int num_cols) {
  ColRange range;

  const int cols_per_process = num_cols / size;
  const int remainder = num_cols % size;

  range.start = (rank * cols_per_process) + std::min(rank, remainder);
  const int cols_for_rank = cols_per_process + (rank < remainder ? 1 : 0);
  range.end = range.start + cols_for_rank;

  return range;
}

int FindOwnerRank(int col, int size, int num_cols) {
  const int cols_per_process = num_cols / size;
  const int remainder = num_cols % size;

  int start = 0;
  for (int proc = 0; proc < size; ++proc) {
    const int cols_for_rank = cols_per_process + (proc < remainder ? 1 : 0);
    const int end = start + cols_for_rank;
    if (col >= start && col < end) {
      return proc;
    }
    start = end;
  }
  return size - 1;
}

std::vector<double> BuildLocalA(const std::vector<double> &global_a, int n, const ColRange &local_cols) {
  const int local_cols_count = local_cols.end - local_cols.start;
  std::vector<double> local_a(static_cast<std::size_t>(n) * static_cast<std::size_t>(local_cols_count));
  for (int row = 0; row < n; ++row) {
    for (int gc = local_cols.start; gc < local_cols.end; ++gc) {
      const int lc = gc - local_cols.start;
      local_a[LocalAt(row, lc, local_cols_count)] = At(global_a, n, row, gc);
    }
  }
  return local_a;
}

int FindPivotRowLocal(const std::vector<double> &local_a, int n, int k, int lc_k, int local_cols_count) {
  int pivot_row = k;
  double best = std::abs(local_a[LocalAt(k, lc_k, local_cols_count)]);
  for (int row = k + 1; row < n; ++row) {
    const double v = std::abs(local_a[LocalAt(row, lc_k, local_cols_count)]);
    if (v > best) {
      best = v;
      pivot_row = row;
    }
  }
  return pivot_row;
}

void SwapRowsLocal(std::vector<double> &local_a, int r1, int r2, int local_cols_count) {
  if (r1 == r2) {
    return;
  }
  for (int lc = 0; lc < local_cols_count; ++lc) {
    std::swap(local_a[LocalAt(r1, lc, local_cols_count)], local_a[LocalAt(r2, lc, local_cols_count)]);
  }
}

int BroadcastPivotRow(int k, const std::vector<double> &local_a, int n, const ColRange &local_cols,
                      int local_cols_count, int owner, int rank) {
  int pivot_row = k;
  if (rank == owner) {
    const int lc_k = k - local_cols.start;
    pivot_row = FindPivotRowLocal(local_a, n, k, lc_k, local_cols_count);
  }
  MPI_Bcast(&pivot_row, 1, MPI_INT, owner, MPI_COMM_WORLD);
  return pivot_row;
}

void ApplyPivotSwapAll(std::vector<double> &local_a, std::vector<double> &b, int k, int pivot_row,
                       int local_cols_count) {
  if (pivot_row == k) {
    return;
  }
  SwapRowsLocal(local_a, k, pivot_row, local_cols_count);
  std::swap(b[static_cast<std::size_t>(k)], b[static_cast<std::size_t>(pivot_row)]);
}

double BroadcastPivotValue(int k, const std::vector<double> &local_a, const ColRange &local_cols, int local_cols_count,
                           int owner, int rank) {
  double pivot = 0.0;
  if (rank == owner) {
    const int lc_k = k - local_cols.start;
    pivot = local_a[LocalAt(k, lc_k, local_cols_count)];
  }
  MPI_Bcast(&pivot, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
  return pivot;
}

std::vector<double> BuildAndBroadcastMultipliers(std::vector<double> &local_a, int n, int k, double pivot,
                                                 const ColRange &local_cols, int local_cols_count, int owner,
                                                 int rank) {
  std::vector<double> multipliers(static_cast<std::size_t>(n - (k + 1)));
  if (rank == owner) {
    const int lc_k = k - local_cols.start;
    for (int row = k + 1; row < n; ++row) {
      const double v = local_a[LocalAt(row, lc_k, local_cols_count)];
      multipliers[static_cast<std::size_t>(row - (k + 1))] = v / pivot;
      local_a[LocalAt(row, lc_k, local_cols_count)] = 0.0;
    }
  }
  if (!multipliers.empty()) {
    MPI_Bcast(multipliers.data(), static_cast<int>(multipliers.size()), MPI_DOUBLE, owner, MPI_COMM_WORLD);
  }
  return multipliers;
}

void EliminateLocalBlock(std::vector<double> &local_a, std::vector<double> &b, int n, int k, const ColRange &local_cols,
                         int local_cols_count, const std::vector<double> &multipliers) {
  for (int row = k + 1; row < n; ++row) {
    const double m = multipliers[static_cast<std::size_t>(row - (k + 1))];
    for (int gc = std::max(k + 1, local_cols.start); gc < local_cols.end; ++gc) {
      const int lc = gc - local_cols.start;
      local_a[LocalAt(row, lc, local_cols_count)] -= m * local_a[LocalAt(k, lc, local_cols_count)];
    }
    b[static_cast<std::size_t>(row)] -= m * b[static_cast<std::size_t>(k)];
  }
}

bool ForwardEliminationMPI(std::vector<double> &local_a, std::vector<double> &b, int n, const ColRange &local_cols,
                           int local_cols_count, int rank, int size) {
  for (int k = 0; k < n; ++k) {
    const int owner = FindOwnerRank(k, size, n);
    const int pivot_row = BroadcastPivotRow(k, local_a, n, local_cols, local_cols_count, owner, rank);
    ApplyPivotSwapAll(local_a, b, k, pivot_row, local_cols_count);
    const double pivot = BroadcastPivotValue(k, local_a, local_cols, local_cols_count, owner, rank);
    if (std::abs(pivot) < kEps) {
      return false;
    }
    const std::vector<double> multipliers =
        BuildAndBroadcastMultipliers(local_a, n, k, pivot, local_cols, local_cols_count, owner, rank);
    EliminateLocalBlock(local_a, b, n, k, local_cols, local_cols_count, multipliers);
  }
  return true;
}

double AllreducePartialSum(double local_value) {
  double sum = 0.0;
  MPI_Allreduce(&local_value, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sum;
}

double ComputeLocalRowDot(const std::vector<double> &local_a, const std::vector<double> &x, int i,
                          const ColRange &local_cols, int local_cols_count) {
  double partial_sum = 0.0;
  for (int gc = std::max(i + 1, local_cols.start); gc < local_cols.end; ++gc) {
    const int lc = gc - local_cols.start;
    partial_sum += local_a[LocalAt(i, lc, local_cols_count)] * x[static_cast<std::size_t>(gc)];
  }
  return partial_sum;
}

double BroadcastDiagonal(const std::vector<double> &local_a, int i, const ColRange &local_cols, int local_cols_count,
                         int owner, int rank) {
  double diag = 0.0;
  if (rank == owner) {
    const int lc = i - local_cols.start;
    diag = local_a[LocalAt(i, lc, local_cols_count)];
  }
  MPI_Bcast(&diag, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
  return diag;
}

bool BackSubstitutionMPI(const std::vector<double> &local_a, std::vector<double> &b, int n, const ColRange &local_cols,
                         int local_cols_count, int rank, int size, std::vector<double> &x) {
  for (int i = n - 1; i >= 0; --i) {
    const double partial_sum = ComputeLocalRowDot(local_a, x, i, local_cols, local_cols_count);
    const double sum = AllreducePartialSum(partial_sum);

    const int owner = FindOwnerRank(i, size, n);
    const double diag = BroadcastDiagonal(local_a, i, local_cols, local_cols_count, owner, rank);
    if (std::abs(diag) < kEps) {
      return false;
    }

    x[static_cast<std::size_t>(i)] = (b[static_cast<std::size_t>(i)] - sum) / diag;
  }
  return true;
}

}  // namespace

CheremkhinAGausVertMPI::CheremkhinAGausVertMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool CheremkhinAGausVertMPI::ValidationImpl() {
  const auto &in = GetInput();
  if (in.n <= 0) {
    return false;
  }
  const auto n = static_cast<std::size_t>(in.n);
  return in.a.size() == n * n && in.b.size() == n;
}

bool CheremkhinAGausVertMPI::PreProcessingImpl() {
  return true;
}

bool CheremkhinAGausVertMPI::RunImpl() {
  const auto &in = GetInput();

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int n = in.n;
  const ColRange local_cols = GetColRange(rank, size, n);
  const int local_cols_count = local_cols.end - local_cols.start;

  std::vector<double> local_a = BuildLocalA(in.a, n, local_cols);
  std::vector<double> b = in.b;
  std::vector<double> x(static_cast<std::size_t>(n), 0.0);

  if (!ForwardEliminationMPI(local_a, b, n, local_cols, local_cols_count, rank, size)) {
    return false;
  }

  if (!BackSubstitutionMPI(local_a, b, n, local_cols, local_cols_count, rank, size, x)) {
    return false;
  }

  GetOutput() = x;
  return true;
}

bool CheremkhinAGausVertMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace cheremkhin_a_gaus_vert
