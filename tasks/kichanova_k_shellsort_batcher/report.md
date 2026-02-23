# Отчёт по лабораторной работе
# Сортировка Шелла с четно-нечетным слиянием Бэтчера

- Student: Кичанова Ксения Константиновна, group 3823Б1ФИ3
- Technology: SEQ | MPI
- Variant: 17

## 1. Introduction
Сортировка Шелла с четно-нечетным слиянием Бэтчера — гибридный алгоритм сортировки, сочетающий эффективность сортировки Шелла и параллельные возможности сети сортировки Бэтчера. 
В контексте параллельного программирования данная задача представляет интерес благодаря возможности распараллеливания операции сортировки больших массивов данных. 
Ожидается, что при использовании технологии MPI произойдёт ускорение по сравнению с последовательной версией.

## 2. Problem Statement
Нужно отсортировать массив целых чисел с использованием комбинации сортировки Шелла и четно-нечетного слияния Бэтчера.
Input: Целое число - размер массива
Output: Целое число - контрольная сумма отсортированного массива

## 3. Baseline Algorithm (Sequential)
Базовый последовательный алгоритм состоит из двух основных этапов:
1. Сортировка массива алгоритмом Шелла с последовательностью Кнута
2. Разделение массива на две части и их слияние с использованием четно-нечетного слияния Бэтчера
Вычислительная сложность: O(N^1.25) - O(N^2)

## 4. Parallelization Scheme
Каждый процесс MPI получает для обработки сегмент исходного массива. Размер сегмента вычисляется с учетом равномерного распределения данных. 
Каждый процесс независимо генерирует свою часть данных и выполняет локальную сортировку с использованием алгоритма Шелла. 
Затем используется упрощенный алгоритм четно-нечетной транспозиционной сортировки: в каждой фазе процессы объединяются в пары согласно четности их рангов. 
На каждом этапе процессы обмениваются данными, выполняют слияние отсортированных массивов и сохраняют нужную часть результата. 
После завершения всех фаз сортировки каждый процесс вычисляет локальную контрольную сумму. 
С помощью операции MPI_Allreduce с суммированием вычисляется глобальная контрольная сумма.

## 5. Implementation Details
- common.hpp - общие типы данных и константы
- ops_seq - последовательная реализация алгоритма
- ops_mpi - параллельная MPI-реализация алгоритма
- tests - functional для проверки корректности и performance для замера скорости

## 6. Experimental Setup
- Аппаратное обеспечение: Intel® Core™ Ultra 5 225U × 14 (12 ядер, 14 логических процессоров, базовая скорость 1,5 ГГц)
- ОЗУ — 32 ГБ 
- Операционная система: Ubuntu 24.04.3 LTS
- Компилятор: g++
- Тип сборки: Release

## 7. Results and Discussion

### 7.1 Correctness
Корректность реализации проверялась тестами, включающие массивы размером 100, 1000, 5000 и 10000 элементов. 
Все функциональные тесты пройдены успешно.

### 7.2 Performance
Сравнение seq и mpi на разных процессах проводилось на массиве состоящем из 10000 элементов.

pipeline:

| Mode        | Count | Time, s   | Speedup | Efficiency  |
|-------------|-------|-----------|---------|-------------|
| seq         | 1     | 0.00098   | 1.00    | N/A         |
| omp         | 2     | 0.00092   | 1.07    | 53.5%       |
| omp         | 4     | 0.00157   | 0.62    | 15.5%       |
| omp         | 8     | 0.00054   | 1.81    | 22.6%       |

task_run:

| Mode        | Count | Time, s   | Speedup | Efficiency  |
|-------------|-------|-----------|---------|-------------|
| seq         | 1     | 0.00105   | 1.00    | N/A         |
| omp         | 2     | 0.00056   | 1.88    | 94.0%       |
| omp         | 4     | 0.00045   | 2.33    | 58.3%       |
| omp         | 8     | 0.00019   | 5.53    | 69.1%       |


## 8. Conclusions
MPI реализация демонстрирует положительное ускорение в режиме task_run на всех конфигурациях. 
Максимальное ускорение 5.53 достигнуто в режиме task_run на 8 процессах. 
В режиме pipeline наблюдается нестабильная производительность с ускорением только на 2 и 8 процессах. 
Эффективность параллелизации в режиме task_run остается высокой. 
Полученные результаты подтверждают эффективность выбранного подхода к распараллеливанию алгоритма сортировки с использованием четно-нечетной транспозиционной сортировки.

## 9. References
1. Лекции и практики курса "Параллельное программирование для кластерных систем"
2. Документация по MPI (стандарт MPI-3.1).

## Appendix
ops_seq.cpp:
```cpp
bool KichanovaKShellsortBatcherSEQ::RunImpl() {
  const InType n = GetInput();

  if (n <= 0) {
    return false;
  }

  std::vector<int> data(static_cast<std::size_t>(n));
  std::mt19937 gen(static_cast<unsigned int>(n));
  std::uniform_int_distribution<int> dist(0, 1000000);

  for (int &v : data) {
    v = dist(gen);
  }

  std::vector<int> expected = data;
  std::sort(expected);

  ShellSort(data);

  const auto mid = data.size() / 2;
  std::vector<int> left(data.begin(), data.begin() + static_cast<std::vector<int>::difference_type>(mid));
  std::vector<int> right(data.begin() + static_cast<std::vector<int>::difference_type>(mid), data.end());
  std::vector<int> merged;
  OddEvenBatcherMerge(left, right, merged);
  data.swap(merged);

  if (!std::is_sorted(data)) {
    return false;
  }
  if (data != expected) {
    return false;
  }

  std::int64_t checksum = std::accumulate(data.begin(), data.end(), static_cast<std::int64_t>(0));
  GetOutput() = static_cast<OutType>(checksum & 0x7FFFFFFF);

  return true;
}

static void KichanovaKShellsortBatcherSEQ::ShellSort(std::vector<int> &arr) {
  const std::size_t n = arr.size();
  if (n < 2) {
    return;
  }

  std::size_t gap = 1;
  while (gap < n / 3) {
    gap = (gap * 3) + 1;
  }

  while (gap > 0) {
    for (std::size_t i = gap; i < n; ++i) {
      const int tmp = arr[i];
      std::size_t j = i;
      while (j >= gap && arr[j - gap] > tmp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = tmp;
    }
    gap = (gap - 1) / 3;
  }
}
```
ops_mpi.cpp:
```cpp
bool KichanovaKShellsortBatcherMPI::RunImpl() {
  const InType n = GetInput();
  if (n <= 0) {
    return false;
  }

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto local_data = GenerateLocalData(n, rank, size);
  ShellSort(local_data);
  PerformOddEvenSort(local_data, rank, size);

  std::int64_t local_checksum = CalculateChecksum(local_data);
  std::int64_t global_checksum = 0;
  MPI_Allreduce(&local_checksum, &global_checksum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<OutType>(global_checksum & 0x7FFFFFFF);
  return true;
}

std::vector<int> KichanovaKShellsortBatcherMPI::GenerateLocalData(InType n, int rank, int size) {
  InType base = n / size;
  InType rem = n % size;
  InType local_n = base + (rank < rem ? 1 : 0);
  
  std::vector<int> local_data(local_n);
  std::mt19937 rng(static_cast<unsigned int>(n));
  std::uniform_int_distribution<int> dist(0, 1000000);
  
  InType offset = 0;
  for (int i = 0; i < rank; ++i) {
    InType proc_n = base + (i < rem ? 1 : 0);
    offset += proc_n;
  }
  
  for (InType i = 0; i < offset; ++i) {
    (void)dist(rng);
  }
  
  for (InType i = 0; i < local_n; ++i) {
    local_data[i] = dist(rng);
  }
  
  return local_data;
}

void KichanovaKShellsortBatcherMPI::PerformOddEvenSort(std::vector<int>& local_data, int rank, int size) {
  for (int phase = 0; phase < size; ++phase) {
    int partner = GetPartner(phase, rank);
    
    if (partner >= 0 && partner < size) {
      ExchangeAndMerge(local_data, partner, rank, 1000 + phase);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

int KichanovaKShellsortBatcherMPI::GetPartner(int phase, int rank) const {
  if (phase % 2 == 0) {
    return (rank % 2 == 0) ? rank + 1 : rank - 1;
  } else {
    return (rank % 2 == 1) ? rank + 1 : rank - 1;
  }
}

std::int64_t KichanovaKShellsortBatcherMPI::CalculateChecksum(const std::vector<int>& data) const {
  std::int64_t checksum = 0;
  for (const auto &val : data) {
    checksum += val;
  }
  return checksum;
}

void KichanovaKShellsortBatcherMPI::ShellSort(std::vector<int> &arr) {
  const std::size_t n = arr.size();
  if (n < 2) {
    return;
  }

  std::size_t gap = 1;
  while (gap < n / 3) {
    gap = (gap * 3) + 1;
  }

  while (gap > 0) {
    for (std::size_t i = gap; i < n; ++i) {
      const int tmp = arr[i];
      std::size_t j = i;
      while (j >= gap && arr[j - gap] > tmp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = tmp;
    }
    gap = (gap - 1) / 3;
  }
}
```