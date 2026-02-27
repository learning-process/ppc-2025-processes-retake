#include "marov_radix_sort_double/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "marov_radix_sort_double/common/include/common.hpp"

namespace marov_radix_sort_double {

namespace {

// Преобразование double в uint64_t для сортировки
uint64_t DoubleToSortableUint64(double val) {
  uint64_t bits = 0;
  std::memcpy(&bits, &val, sizeof(double));
  // Инвертируем биты для отрицательных чисел
  if ((bits >> 63) != 0) {
    bits = ~bits;
  } else {
    bits |= (1ULL << 63);
  }
  return bits;
}

// Обратное преобразование
double SortableUint64ToDouble(uint64_t bits) {
  // Восстанавливаем исходное представление
  if ((bits >> 63) != 0) {
    bits &= ~(1ULL << 63);
  } else {
    bits = ~bits;
  }
  double val = 0;
  std::memcpy(&val, &bits, sizeof(double));
  return val;
}

// Поразрядная сортировка для массива double
void RadixSortDoubles(std::vector<double>& data) {
  if (data.size() <= 1) {
    return;
  }

  std::vector<uint64_t> keys(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    keys[i] = DoubleToSortableUint64(data[i]);
  }

  const int kRadix = 256;
  std::vector<uint64_t> temp(data.size());

  for (int shift = 0; shift < 64; shift += 8) {
    std::vector<size_t> count(kRadix + 1, 0);

    for (uint64_t key : keys) {
      uint8_t digit = (key >> shift) & 0xFF;
      ++count[digit + 1];
    }

    for (int i = 0; i < kRadix; ++i) {
      count[i + 1] += count[i];
    }

    for (uint64_t key : keys) {
      uint8_t digit = (key >> shift) & 0xFF;
      size_t pos = count[digit];
      temp[pos] = key;
      ++count[digit];
    }

    keys.swap(temp);
  }

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = SortableUint64ToDouble(keys[i]);
  }
}

// Слияние двух отсортированных массивов
std::vector<double> MergeSorted(const std::vector<double>& a,
                                const std::vector<double>& b) {
  std::vector<double> result;
  result.reserve(a.size() + b.size());
  size_t i = 0;
  size_t j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] < b[j]) {
      result.push_back(a[i++]);
    } else {
      result.push_back(b[j++]);
    }
  }
  while (i < a.size()) {
    result.push_back(a[i++]);
  }
  while (j < b.size()) {
    result.push_back(b[j++]);
  }
  return result;
}

}  // namespace

MarovRadixSortDoubleMPI::MarovRadixSortDoubleMPI(const InType& in) {
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_size_);

  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarovRadixSortDoubleMPI::ValidationImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  return initialized != 0;
}

bool MarovRadixSortDoubleMPI::PreProcessingImpl() {
  return true;
}

bool MarovRadixSortDoubleMPI::RunImpl() {
  const auto& input = GetInput();
  int n = static_cast<int>(input.size());

  // Рассылка размера массива
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Вычисление распределения данных
  std::vector<int> counts(proc_size_);
  std::vector<int> displs(proc_size_);

  int base = n / proc_size_;
  int rem = n % proc_size_;
  int offset = 0;

  for (int i = 0; i < proc_size_; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }

  // Локальный массив для каждого процесса
  std::vector<double> local_data(counts[proc_rank_]);

  // Рассылка данных
  MPI_Scatterv(proc_rank_ == 0 ? input.data() : nullptr, counts.data(),
               displs.data(), MPI_DOUBLE, local_data.data(),
               counts[proc_rank_], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Локальная сортировка
  RadixSortDoubles(local_data);

  // Сбор и слияние результатов на процессе 0
  std::vector<double> result;
  if (proc_rank_ == 0) {
    result = std::move(local_data);

    // Получение данных от остальных процессов и слияние
    for (int proc = 1; proc < proc_size_; ++proc) {
      int recv_count = 0;
      MPI_Recv(&recv_count, 1, MPI_INT, proc, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      std::vector<double> part(recv_count);
      MPI_Recv(part.data(), recv_count, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      result = MergeSorted(result, part);
    }

    GetOutput() = result;
  } else {
    // Отправка данных процессу 0
    int send_count = static_cast<int>(local_data.size());
    MPI_Send(&send_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(local_data.data(), send_count, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
  }

  return true;
}

bool MarovRadixSortDoubleMPI::PostProcessingImpl() {
  return true;
}

}  // namespace marov_radix_sort_double
