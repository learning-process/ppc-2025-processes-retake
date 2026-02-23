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

  std::vector<double> local_a(static_cast<std::size_t>(n) * static_cast<std::size_t>(local_cols_count));
  for (int row = 0; row < n; ++row) {
    for (int gc = local_cols.start; gc < local_cols.end; ++gc) {
      const int lc = gc - local_cols.start;
      local_a[(static_cast<std::size_t>(row) * static_cast<std::size_t>(local_cols_count)) +
              static_cast<std::size_t>(lc)] = At(in.a, n, row, gc);
    }
  }

  std::vector<double> b = in.b;  
  std::vector<double> x(static_cast<std::size_t>(n), 0.0);

  for (int k = 0; k < n; ++k) {
    const int owner = FindOwnerRank(k, size, n);

    int pivot_row = k;
    double pivot = 0.0;
    std::vector<double> multipliers;

    if (rank == owner) {
      const int lc_k = k - local_cols.start;
      pivot_row = k;
      double best = std::abs(local_a[(static_cast<std::size_t>(k) * static_cast<std::size_t>(local_cols_count)) +
                                     static_cast<std::size_t>(lc_k)]);
      for (int row = k + 1; row < n; ++row) {
        const double v = std::abs(local_a[(static_cast<std::size_t>(row) * static_cast<std::size_t>(local_cols_count)) +
                                          static_cast<std::size_t>(lc_k)]);
        if (v > best) {
          best = v;
          pivot_row = row;
        }
      }
    }

    MPI_Bcast(&pivot_row, 1, MPI_INT, owner, MPI_COMM_WORLD);

    if (pivot_row != k) {
      for (int lc = 0; lc < local_cols_count; ++lc) {
        std::swap(local_a[(static_cast<std::size_t>(k) * static_cast<std::size_t>(local_cols_count)) +
                          static_cast<std::size_t>(lc)],
                  local_a[(static_cast<std::size_t>(pivot_row) * static_cast<std::size_t>(local_cols_count)) +
                          static_cast<std::size_t>(lc)]);
      }
      std::swap(b[static_cast<std::size_t>(k)], b[static_cast<std::size_t>(pivot_row)]);
    }

    if (rank == owner) {
      const int lc_k = k - local_cols.start;
      pivot = local_a[(static_cast<std::size_t>(k) * static_cast<std::size_t>(local_cols_count)) +
                      static_cast<std::size_t>(lc_k)];
    }
    MPI_Bcast(&pivot, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    if (std::abs(pivot) < kEps) {
      return false;
    }

    if (rank == owner) {
      const int lc_k = k - local_cols.start;
      multipliers.resize(static_cast<std::size_t>(n - (k + 1)));
      for (int row = k + 1; row < n; ++row) {
        const double v = local_a[(static_cast<std::size_t>(row) * static_cast<std::size_t>(local_cols_count)) +
                                 static_cast<std::size_t>(lc_k)];
        multipliers[static_cast<std::size_t>(row - (k + 1))] = v / pivot;
        local_a[(static_cast<std::size_t>(row) * static_cast<std::size_t>(local_cols_count)) +
                static_cast<std::size_t>(lc_k)] = 0.0;
      }
    } else {
      multipliers.resize(static_cast<std::size_t>(n - (k + 1)));
    }

    if (!multipliers.empty()) {
      MPI_Bcast(multipliers.data(), static_cast<int>(multipliers.size()), MPI_DOUBLE, owner, MPI_COMM_WORLD);
    }

    for (int row = k + 1; row < n; ++row) {
      const double m = multipliers[static_cast<std::size_t>(row - (k + 1))];
      for (int gc = std::max(k + 1, local_cols.start); gc < local_cols.end; ++gc) {
        const int lc = gc - local_cols.start;
        local_a[(static_cast<std::size_t>(row) * static_cast<std::size_t>(local_cols_count)) +
                static_cast<std::size_t>(lc)] -=
            m * local_a[(static_cast<std::size_t>(k) * static_cast<std::size_t>(local_cols_count)) +
                        static_cast<std::size_t>(lc)];
      }
      b[static_cast<std::size_t>(row)] -= m * b[static_cast<std::size_t>(k)];
    }
  }

  for (int i = n - 1; i >= 0; --i) {
    double partial_sum = 0.0;
    for (int gc = std::max(i + 1, local_cols.start); gc < local_cols.end; ++gc) {
      const int lc = gc - local_cols.start;
      partial_sum += local_a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(local_cols_count)) +
                             static_cast<std::size_t>(lc)] *
                     x[static_cast<std::size_t>(gc)];
    }

    double sum = 0.0;
    MPI_Allreduce(&partial_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    const int owner = FindOwnerRank(i, size, n);
    double diag = 0.0;
    if (rank == owner) {
      const int lc = i - local_cols.start;
      diag = local_a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(local_cols_count)) +
                     static_cast<std::size_t>(lc)];
    }
    MPI_Bcast(&diag, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    if (std::abs(diag) < kEps) {
      return false;
    }

    const double xi = (b[static_cast<std::size_t>(i)] - sum) / diag;
    x[static_cast<std::size_t>(i)] = xi;
  }

  GetOutput() = x;
  return true;
}

bool CheremkhinAGausVertMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace cheremkhin_a_gaus_vert
