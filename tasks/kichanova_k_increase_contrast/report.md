# Отчёт по лабораторной работе

## Повышение контраста

- Student: Кичанова Ксения Константиновна, group 3823Б1ФИ3
- Technology: SEQ | MPI
- Variant: 23

## 1. Introduction

Повышение контраста изображения — базовая операция в обработке изображений,
направленная на увеличение разницы между яркостными характеристиками объектов.
Данная задача относится к классу Data Parallel, так как независимые операции выполняются над разными частями данных (пикселями).
Ожидается, что использование технологии MPI позволит добиться ускорения обработки за счет 
распределения вычислительной нагрузки между несколькими процессами, особенно для изображений большого размера.

## 2. Problem Statement

Необходимо применить алгоритм линейного контрастирования к RGB-изображению.
Input: Структура Image, содержащая одномерный массив пикселей (std::vector<uint8_t> pixels)
и метаданные (ширина, высота, количество каналов).
Для задачи используется 3 канала (RGB).
Output: Структура Image того же размера с увеличенным контрастом.

## 3. Baseline Algorithm (Sequential)

Базовый последовательный алгоритм состоит из двух проходов по всем пикселям изображения:
Проход 1: Поиск глобальных минимальных и максимальных значений для каждого из трёх цветовых каналов (R, G, B).
Проход 2: Применение формулы линейного контрастирования к каждому пикселю и каждому каналу 
с использованием найденных глобальных min и max.
Вычислительная сложность: O(N)

## 4. Parallelization Scheme

Изображение разделяется по горизонтальным строкам между доступными процессами MPI.
Каждый процесс получает для обработки свой сегмент изображения.
Каждый процесс находит минимальные и максимальные значения яркости для каждого канала (R, G, B) в своём сегменте.
С помощью операций MPI_Allreduce с операциями MPI_MIN и MPI_MAX находятся глобальные минимумы и максимумы для каждого канала.
Каждый процесс независимо применяет формулу линейного контрастирования ко всем пикселям своего сегмента,
используя вычисленные глобальные значения.С помощью MPI_Allgatherv все процессы обмениваются обработанными сегментами, 
и каждый процесс получает полное итоговое изображение.

## 5. Implementation Details

- common.hpp - общие типы данных и структура Image для представления изображения
- ops_seq - последовательная реализация алгоритма
- ops_mpi - параллельная MPI-реализация алгоритма
- tests - functional для проверки корректности и performance для замера скорости.

## 6. Experimental Setup

- Аппаратное обеспечение: Intel® Core™ Ultra 5 225U × 14 (12 ядер, 14 логических процессоров, базовая скорость 1,5 ГГц)
- ОЗУ — 32 ГБ
- Операционная система: Ubuntu 24.04.3 LTS
- Компилятор: g++
- Тип сборки: Release

## 7. Results and Discussion

### 7.1 Correctness

Корректность реализации проверялась набором функциональных тестов: маленькое однородное изображение (4×4 пикселя), 
изображение с градиентом (8×8 пикселей), краевой случай для MPI (1×3 пикселя), реальное изображение из файла.
Все 8 тестов (4 для MPI и 4 для последовательной версии) успешно пройдены.
Результаты параллельной реализации полностью совпали с результатами последовательной.

### 7.2 Performance

pipeline:

| Mode        | Count | Time, s   | Speedup | Efficiency  |
|-------------|-------|-----------|---------|-------------|
| seq         | 1     | 0.21347   | 1.00    | N/A         |
| omp         | 2     | 0.12488   | 1.71    | 85.5%       |
| omp         | 4     | 0.11009   | 1.94    | 48.5%       |
| omp         | 8     | 0.13576   | 1.57    | 19.6%       |

task_run:

| Mode        | Count | Time, s   | Speedup | Efficiency  |
|-------------|-------|-----------|---------|-------------|
| seq         | 1     | 0.21388   | 1.00    | N/A         |
| omp         | 2     | 0.12458   | 1.72    | 86.0%       |
| omp         | 4     | 0.11442   | 1.87    | 46.8%       |
| omp         | 8     | 0.13556   | 1.58    | 19.8%       |

## 8. Conclusions

MPI реализация алгоритма увеличения контраста демонстрирует положительное ускорение на всех тестируемых конфигурациях.
Максимальное ускорение 1.94 достигнуто в режиме pipeline при использовании 4 процессов.
На 8 процессах ускорение несколько снижается (до 1.58), что связано с возросшими накладными расходами на коммуникацию.

## 9. References

1. Лекции и практики курса "Параллельное программирование для кластерных систем"
2. Документация по MPI (стандарт MPI-3.1).
3. Лекции и практики курса "Компьюьтерная графика"

## Appendix

ops_seq.cpp:

bool KichanovaKIncreaseContrastSEQ::RunImpl() {
  const auto& input = GetInput();
  auto& output = GetOutput();

  const int width = input.width;
  const int height = input.height;
  const int channels = 3;
  const size_t total_pixels = width * height;

  uint8_t min_r = 255, max_r = 0;
  uint8_t min_g = 255, max_g = 0;
  uint8_t min_b = 255, max_b = 0;

  for (size_t i = 0; i < total_pixels; ++i) {
    size_t idx = i * channels;

    uint8_t r = input.pixels[idx];
    uint8_t g = input.pixels[idx + 1];
    uint8_t b = input.pixels[idx + 2];

    if (r < min_r) min_r = r;
    if (r > max_r) max_r = r;
    if (g < min_g) min_g = g;
    if (g > max_g) max_g = g;
    if (b < min_b) min_b = b;
    if (b > max_b) max_b = b;
  }

  float scale_r = 0.0f, scale_g = 0.0f, scale_b = 0.0f;

  if (max_r > min_r) {
    scale_r = 255.0f / (max_r - min_r);
  }

  if (max_g > min_g) {
    scale_g = 255.0f / (max_g - min_g);
  }

  if (max_b > min_b) {
    scale_b = 255.0f / (max_b - min_b);
  }

  for (size_t i = 0; i < total_pixels; ++i) {
    size_t idx = i * channels;

    uint8_t r = input.pixels[idx];
    uint8_t g = input.pixels[idx + 1];
    uint8_t b = input.pixels[idx + 2];

    if (max_r > min_r) {
      float new_r = (r - min_r) * scale_r;
      output.pixels[idx] = static_cast<uint8_t>(std::clamp(new_r, 0.0f, 255.0f));
    } else {
      output.pixels[idx] = r;
    }

    if (max_g > min_g) {
      float new_g = (g - min_g) * scale_g;
      output.pixels[idx + 1] = static_cast<uint8_t>(std::clamp(new_g, 0.0f, 255.0f));
    } else {
      output.pixels[idx + 1] = g;
    }

    if (max_b > min_b) {
      float new_b = (b - min_b) * scale_b;
      output.pixels[idx + 2] = static_cast<uint8_t>(std::clamp(new_b, 0.0f, 255.0f));
    } else {
      output.pixels[idx + 2] = b;
    }
  }

  return true;
}

ops_mpi.cpp:

bool KichanovaKIncreaseContrastMPI::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  const int width = input.width;
  const int height = input.height;
  const int channels = 3;
  const int row_size = width * channels;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows_per_process = height / size;
  int remainder = height % size;
  int start_row = rank * rows_per_process + std::min(rank, remainder);
  int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);
  int local_rows = end_row - start_row;

  uint8_t local_min[3] = {255, 255, 255};
  uint8_t local_max[3] = {0, 0, 0};

  for (int row = start_row; row < end_row; ++row) {
    for (int col = 0; col < width; ++col) {
      size_t idx = (row *width + col)* channels;
      for (int c = 0; c < 3; ++c) {
        uint8_t val = input.pixels[idx + c];
        if (val < local_min[c]) local_min[c] = val;
        if (val > local_max[c]) local_max[c] = val;
      }
    }
  }

  uint8_t global_min[3], global_max[3];
  MPI_Allreduce(local_min, global_min, 3, MPI_UINT8_T, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(local_max, global_max, 3, MPI_UINT8_T, MPI_MAX, MPI_COMM_WORLD);

  float scale[3];
  bool need_scale[3];
  for (int c = 0; c < 3; ++c) {
    if (global_max[c] > global_min[c]) {
      scale[c] = 255.0f / (global_max[c] - global_min[c]);
      need_scale[c] = true;
    } else {
      scale[c] = 0.0f;
      need_scale[c] = false;
    }
  }

  std::vector<uint8_t> local_output(local_rows *row_size);
  for (int i = 0; i < local_rows; ++i) {
    int global_row = start_row + i;
    for (int col = 0; col < width; ++col) {
      size_t in_idx = (global_row* width + col) *channels;
      size_t out_idx = (i* width + col) *channels;
      for (int c = 0; c < 3; ++c) {
        uint8_t val = input.pixels[in_idx + c];
        if (need_scale[c]) {
          float new_val = (val - global_min[c])* scale[c];
          local_output[out_idx + c] = static_cast<uint8_t>(std::clamp(new_val, 0.0f, 255.0f));
        } else {
          local_output[out_idx + c] = val;
        }
      }
    }
  }

  std::vector<`int`> recv_counts(size);
  std::vector<`int`> displs(size);
  for (int i = 0; i < size; ++i) {
    int i_start_row = i *rows_per_process + std::min(i, remainder);
    int i_end_row = i_start_row + rows_per_process + (i < remainder ? 1 : 0);
    int i_rows = i_end_row - i_start_row;
    recv_counts[i] = i_rows* row_size;
    displs[i] = i_start_row * row_size;
  }

  MPI_Allgatherv(local_output.data(), local_rows * row_size, MPI_UINT8_T,
  output.pixels.data(), recv_counts.data(), displs.data(), MPI_UINT8_T, MPI_COMM_WORLD);

  return true;
}
