#pragma once

#include <cstddef>
#include <vector>

#include "task/include/task.hpp"
#include "zyuzin_n_multiplication_matrix_horiz/common/include/common.hpp"

namespace zyuzin_n_multiplication_matrix_horiz {

class ZyuzinNMultiplicationMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZyuzinNMultiplicationMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void BroadcastMatricesInfo(int rank, size_t &rows_a, size_t &cols_a, size_t &rows_b, size_t &cols_b,
                             std::vector<double> &matrix_b_flat);
  void ScatterMatrixA(int rank, int size, size_t rows_a, size_t cols_a, std::vector<double> &matrix_a_flat,
                      std::vector<double> &local_a_flat, int &actual_local_rows);
  static void ComputeLocalMultiplication(const std::vector<double> &local_a_flat,
                                         const std::vector<double> &matrix_b_flat,
                                         std::vector<double> &local_result_flat, int actual_local_rows, size_t cols_a,
                                         size_t cols_b);
  void GatherAndConvertResults(int size, size_t rows_a, size_t cols_b, int actual_local_rows,
                               const std::vector<double> &local_result_flat);
};

}  // namespace zyuzin_n_multiplication_matrix_horiz
