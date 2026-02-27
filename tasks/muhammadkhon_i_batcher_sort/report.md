## Введение

Сортировка массивов является одной из базовых задач алгоритмики и
высокопроизводительных вычислений. При обработке больших объемов
данных последовательные алгоритмы становятся узким местом по времени
выполнения.

Целью данной работы является разработка и исследование параллельной
версии сортировки на основе сети Батчера с использованием технологии
MPI, а также сравнение её производительности с последовательной
реализацией.

## Постановка задачи

Входные данные:

- Массив целых чисел произвольного размера (включая отрицательные
значения).

Выходные данные:

- Отсортированный по возрастанию массив.

Ограничения:

- Использование MPI для распараллеливания.
- Корректная обработка отрицательных чисел.
- Поддержка неравномерного распределения данных между процессами.
- Сохранение корректности при любых размерах массива.

# Описание алгоритма

Последовательная версия состоит из двух этапов:

Локальная сортировка — Radix Sort (LSD)

Для каждого блока данных используется поразрядная сортировка:

- Разделение чисел на положительные и отрицательные.
- Отрицательные числа инвертируются по модулю.
- Выполняется сортировка по основанию 10.
- Отрицательные элементы разворачиваются обратно.

Слияние

После локальной сортировки блоков выполняется слияние с
использованием сортировочной сети Батчера:

- Последовательные фазы с уменьшением шага k.
- Попарные сравнения и обмен элементов.
- Гарантированное получение полностью отсортированного массива.

собственно сам код:

``` C++
bool MuhammadkhonIBatcherSortSEQ::RunImpl() {
  if (GetInput().empty()) {
    GetOutput() = InType();
    return true;
  }
  std::vector<int> data = GetInput();
  for (std::size_t i = 0; i < data.size(); i += kBlockSize) {
    std::size_t current_size = std::min(kBlockSize, data.size() - i);
    std::vector<int> block(data.begin() + static_cast<std::ptrdiff_t>
    (i),
                           data.begin() + static_cast<std::ptrdiff_t>
                           (i + current_size));
    RadixSortLSD(block);
    std::ranges::copy(block, data.begin() + 
    static_cast<std::ptrdiff_t>(i));
  }
  BatcherMergeNetwork(data);
  GetOutput() = std::move(data);
  return true;
}
```

# Схема распараллеливания (MPI)

Декомпозиция данных

Используется блочное распределение:

- Массив делится на size частей.
- Применяется MPI_Scatterv для распределения.
- Учитывается остаток.

Функция CalculateDistribution вычисляет:

- counts — количество элементов на процесс,
- displs — смещения.

Локальная сортировка

Каждый процесс:

- Поучает свою часть массива.
- Выполянет RedixSortLSD() локально.

Сортировка по сети Батчера

Реализована в двух фазах:

Фаза 1 — BatcherNetworkPhase

- Процессы обмениваются данными через MPI_Sendrecv.
- После обмена выполняется слияние:
  - Младший процесс оставляет меньшие элементы.
  - Старший — большие.
- Между шагами используется MPI_Barrier.

Фаза 2 — BatcherStabilizationPhase

- Четно-нечетные шаги.
- Соседние процессы обмениваются данными.
- Максимум 100 итераций для стабилизации.

Каждый процесс получает свой финальный отсортированный массив.

``` C++
bool MuhammadkhonIBatcherSortMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int n = (rank == 0) ? static_cast<int>(GetInput().size()) : 0;
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (n == 0) {
    if (rank == 0) {
      GetOutput() = InType();
    }
    return true;
  }
  auto [counts, displs, local_size] = CalculateDistribution(n, size, rank);
  std::vector<int> local(local_size);
  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, counts.data(), displs.data(), MPI_INT, local.data(),
               static_cast<int>(local_size), MPI_INT, 0, MPI_COMM_WORLD);
  if (!local.empty()) {
    RadixSortLSD(local);
  }
  BatcherNetworkPhase(local, rank, size, counts);
  BatcherStabilizationPhase(local, rank, size, counts);
  std::vector<int> res(static_cast<std::size_t>(n));
  MPI_Allgatherv(local.data(), static_cast<int>(local.size()), MPI_INT, res.data(), counts.data(), displs.data(),
                 MPI_INT, MPI_COMM_WORLD);
  GetOutput() = std::move(res);
  return true;
}
```

# Экперементальные результаты

Окружение

- ОС: Linux(Fedora)
- Компилятор: g++
- Сборка: Release
- MPI: OpenMPI

Результаты производительности

| Версия | Режим    | Время (сек) |
| ------ | -------- | ----------- |
| SEQ    | pipeline | 0.9749      |
| SEQ    | task_run | 0.9751      |
| MPI    | pipeline | 0.01539     |
| MPI    | task_run | 0.01545     |

# Выводы

В ходе работы была реализована параллельная сортировка на основе
сети Батчера с использованием MPI.

Что удалось:

- Реализовать корректное распределение данных.
- Обеспечить правильную обработку отрицательных чисел.
- Добиться значительного ускорения по сравнению с последовательной
версией.
- Реализовать стабильную работу алгоритма.

В целом параллельная версия показала высокую производительность и
хорошую эффективность.
