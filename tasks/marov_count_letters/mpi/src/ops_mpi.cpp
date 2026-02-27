#include "marov_count_letters/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cctype>
#include <string>
#include <vector>

#include "marov_count_letters/common/include/common.hpp"

namespace marov_count_letters {

MarovCountLettersMPI::MarovCountLettersMPI(const InType &in) {
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_size_);

  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MarovCountLettersMPI::ValidationImpl() {
  return true;
}

bool MarovCountLettersMPI::PreProcessingImpl() {
  return true;
}

bool MarovCountLettersMPI::RunImpl() {
  std::string input_str;
  int str_len = 0;

  if (proc_rank_ == 0) {
    input_str = GetInput();
    str_len = static_cast<int>(input_str.size());
  }

  // Рассылка длины строки всем процессам
  MPI_Bcast(&str_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Разделение данных между процессами
  const int base = str_len / proc_size_;
  const int rem = str_len % proc_size_;

  std::vector<int> send_counts(proc_size_);
  std::vector<int> displs(proc_size_);

  if (proc_rank_ == 0) {
    int offset = 0;
    for (int i = 0; i < proc_size_; ++i) {
      send_counts[i] = base + (i < rem ? 1 : 0);
      displs[i] = offset;
      offset += send_counts[i];
    }
  }

  MPI_Bcast(send_counts.data(), proc_size_, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), proc_size_, MPI_INT, 0, MPI_COMM_WORLD);

  const int local_size = base + (proc_rank_ < rem ? 1 : 0);
  std::vector<char> local_data(local_size);

  MPI_Scatterv(proc_rank_ == 0 ? const_cast<char *>(input_str.data()) : nullptr,
               send_counts.data(), displs.data(), MPI_CHAR,
               local_data.data(), local_size, MPI_CHAR, 0, MPI_COMM_WORLD);

  // Подсчет буквенных символов локально
  int local_count = 0;
  for (int i = 0; i < local_size; ++i) {
    if (std::isalpha(static_cast<unsigned char>(local_data[i]))) {
      local_count++;
    }
  }

  // Редукция - сумма всех локальных подсчетов
  int global_count = 0;
  MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (proc_rank_ == 0) {
    GetOutput() = global_count;
  }

  return true;
}

bool MarovCountLettersMPI::PostProcessingImpl() {
  return true;
}

}  // namespace marov_count_letters
