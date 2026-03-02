#pragma once

#include <mpi.h>

#include <vector>

#include "klimov_m_shell_odd_even_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace klimov_m_shell_odd_even_merge {

class ShellBatcherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ShellBatcherMPI(const InputType &input);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

void ShellSortLocal(std::vector<int> &data);
std::vector<int> MergeEvenLeft(std::vector<int> &left, std::vector<int> &right, int chunk, int rank, MPI_Comm comm);
std::vector<int> MergeOddRight(std::vector<int> &left, std::vector<int> &right, int chunk, int rank, MPI_Comm comm);
void ExchangeWithRight(int rank, std::vector<int> &chunk, int chunk_size, MPI_Comm comm);
void ExchangeWithLeft(int rank, std::vector<int> &chunk, int chunk_size, MPI_Comm comm);
void EvenStep(int rank, int procs, std::vector<int> &chunk, int chunk_size, MPI_Comm comm);
void OddStep(int rank, int procs, std::vector<int> &chunk, int chunk_size, MPI_Comm comm);

}  // namespace klimov_m_shell_odd_even_merge
