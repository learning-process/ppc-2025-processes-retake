#include "salykina_a_sparse_matrix_mult/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "salykina_a_sparse_matrix_mult/common/include/common.hpp"

namespace salykina_a_sparse_matrix_mult {

SalykinaASparseMatrixMultSEQ::SalykinaASparseMatrixMultSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = SparseMatrixCRS{};
}

bool SalykinaASparseMatrixMultSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.matrix_a.num_cols != input.matrix_b.num_rows) {
    return false;
  }
  if (input.matrix_a.row_ptr.size() != static_cast<size_t>(static_cast<size_t>(input.matrix_a.num_rows) + 1U)) {
    return false;
  }
  if (input.matrix_b.row_ptr.size() != static_cast<size_t>(static_cast<size_t>(input.matrix_b.num_rows) + 1U)) {
    return false;
  }
  if (input.matrix_a.values.size() != input.matrix_a.col_indices.size()) {
    return false;
  }
  if (input.matrix_b.values.size() != input.matrix_b.col_indices.size()) {
    return false;
  }

  return true;
}

bool SalykinaASparseMatrixMultSEQ::PreProcessingImpl() {
  const auto &input = GetInput();

  auto &output = GetOutput();
  output.num_rows = input.matrix_a.num_rows;
  output.num_cols = input.matrix_b.num_cols;
  output.row_ptr.resize(output.num_rows + 1, 0);

  return true;
}

bool SalykinaASparseMatrixMultSEQ::RunImpl() {
  const auto &input = GetInput();
  const auto &a = input.matrix_a;
  const auto &b = input.matrix_b;

  auto &output = GetOutput();
  std::vector<double> row_result(output.num_cols, 0.0);

  for (int i = 0; i < output.num_rows; i++) {
    std::ranges::fill(row_result, 0.0);
    int row_start = a.row_ptr[i];
    int row_end = a.row_ptr[i + 1];

    for (int k = row_start; k < row_end; k++) {
      int col_a = a.col_indices[k];
      double val_a = a.values[k];

      int b_row_start = b.row_ptr[col_a];
      int b_row_end = b.row_ptr[col_a + 1];

      for (int j = b_row_start; j < b_row_end; j++) {
        int col_b = b.col_indices[j];
        double val_b = b.values[j];
        row_result[col_b] += val_a * val_b;
      }
    }

    for (int j = 0; j < output.num_cols; j++) {
      if (row_result[j] != 0.0) {
        output.values.push_back(row_result[j]);
        output.col_indices.push_back(j);
        output.row_ptr[i + 1]++;
      }
    }
  }

  for (int i = 0; i < output.num_rows; i++) {
    output.row_ptr[i + 1] += output.row_ptr[i];
  }

  return true;
}

bool SalykinaASparseMatrixMultSEQ::PostProcessingImpl() {
  auto &output = GetOutput();

  if (output.values.size() != output.col_indices.size()) {
    return false;
  }
  if (output.row_ptr.size() != static_cast<size_t>(static_cast<size_t>(output.num_rows) + 1U)) {
    return false;
  }

  return true;
}

}  // namespace salykina_a_sparse_matrix_mult
