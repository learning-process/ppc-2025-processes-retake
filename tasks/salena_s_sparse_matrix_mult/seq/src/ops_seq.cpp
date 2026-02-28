#include "salena_s_sparse_matrix_mult/seq/include/ops_seq.hpp"
#include <vector>
#include <algorithm>

namespace salena_s_sparse_matrix_mult {

SparseMatrixMultSeq::SparseMatrixMultSeq(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SparseMatrixMultSeq::ValidationImpl() {
  const auto& A = GetInput().A;
  const auto& B = GetInput().B;
  return (A.cols == B.rows) && (A.rows > 0) && (B.cols > 0);
}

bool SparseMatrixMultSeq::PreProcessingImpl() {
  const auto& A = GetInput().A;
  const auto& B = GetInput().B;
  GetOutput().rows = A.rows;
  GetOutput().cols = B.cols;
  GetOutput().row_ptr.assign(A.rows + 1, 0);
  return true;
}

bool SparseMatrixMultSeq::RunImpl() {
  const auto& A = GetInput().A;
  const auto& B = GetInput().B;
  auto& C = GetOutput();

  C.row_ptr[0] = 0;
  std::vector<int> marker(B.cols, -1);
  std::vector<double> temp_values(B.cols, 0.0);

  for (int i = 0; i < A.rows; ++i) {
    int row_nz = 0;
    std::vector<int> current_row_cols;

    for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
      int a_col = A.col_indices[j];
      double a_val = A.values[j];

      for (int k = B.row_ptr[a_col]; k < B.row_ptr[a_col + 1]; ++k) {
        int b_col = B.col_indices[k];
        double b_val = B.values[k];

        if (marker[b_col] != i) {
          marker[b_col] = i;
          current_row_cols.push_back(b_col);
          temp_values[b_col] = a_val * b_val;
        } else {
          temp_values[b_col] += a_val * b_val;
        }
      }
    }

    std::sort(current_row_cols.begin(), current_row_cols.end());
    for (int col : current_row_cols) {
      if (temp_values[col] != 0.0) {
        C.values.push_back(temp_values[col]);
        C.col_indices.push_back(col);
        row_nz++;
      }
    }
    C.row_ptr[i + 1] = C.row_ptr[i] + row_nz;
  }
  return true;
}

bool SparseMatrixMultSeq::PostProcessingImpl() {
  return true;
}

}  // namespace salena_s_sparse_matrix_mult