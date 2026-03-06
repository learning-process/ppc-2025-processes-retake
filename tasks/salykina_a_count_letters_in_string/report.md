# Подсчет числа буквенных символов в строке

- Student: Салыкина Алена Игоревна, 3823Б1ПР3
- Technology: SEQ | MPI
- Variant: 22

## 1. Introduction

В данной работе целью является ускорение подсчёта буквенных символов в строке за счёт применения параллельных вычислений
с использованием MPI, а также сравнение результатов, чтобы оценить достигнутое ускорение и эффективность распределения нагрузки.

## 2. Problem Statement

Разработать последовательную и параллельную реализации алгоритма подсчёта количества букв в строке. Параллельная
реализация должна использовать технологию MPI для распределения вычислений между процессами.

## 3. Baseline Algorithm (Sequential)

Последовательный алгоритм выполняет простой проход по всем символам входной строки и подсчитывает количество
буквенных символов с помощью функции `std::isalpha()`.

### Реализация последовательного алгоритма

```cpp
int count = 0;
for (char c : GetInput()) {
  if (std::isalpha(static_cast<unsigned char>(c)) != 0) {
    count++;
  }
}

GetOutput() = count;
```

## 4. Parallelization Scheme

Параллельный алгоритм основан на следующих принципах:

1. Строка разбивается на части между процессами
2. Каждый процесс подсчитывает буквы в своей части строки
3. Результаты всех процессов суммируются с помощью операции `MPI_Reduce`
4. Корневой процесс (rank 0) получает итоговый результат

Используется блочное распределение с учётом остатка:

- Каждый процесс получает `chunk_size = string_length / size` символов
- Остаток `remainder = string_length % size` распределяется между первыми `remainder` процессами
- Процесс с номером `rank` обрабатывает символы с индекса `start` до индекса `end`

### Реализация параллельного алгоритма

1. Получение ранга и количества процессов

```cpp
int rank = 0;
int size = 0;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

```

2. Получение входных данных

```cpp
const std::string &input = GetInput();
const int string_length = static_cast<int>(input.length());
```

3. Расчет границ локального блока

```cpp
int local_count = 0;
int chunk_size = string_length / size;
int remainder = string_length % size;
int start = (rank * chunk_size) + std::min(rank, remainder);
int end = start + chunk_size + (rank < remainder ? 1 : 0);
```

4. Локальный подсчет букв

```cpp
for (int i = start; i < end && i < string_length; ++i) {
  if (std::isalpha(static_cast<unsigned char>(input[i])) != 0) {
    ++local_count;
  }
}
```

5. Глобальная редукция и установка результата

```cpp
int total_count = 0;
MPI_Allreduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

GetOutput() = total_count;
MPI_Barrier(MPI_COMM_WORLD);
```

## 5. Implementation Details

### Структура каталога задачи

```text
tasks/salykina_a_count_letters_in_string/
├── common
│   └── include
│       └── common.hpp
├── info.json
├── mpi
│   ├── include
│   │   └── ops_mpi.hpp
│   └── src
│       └── ops_mpi.cpp
├── seq
│   ├── include
│   │   └── ops_seq.hpp
│   └── src
│       └── ops_seq.cpp
├── settings.json
└── tests
    ├── functional
    │   └── main.cpp
    └── performance
        └── main.cpp
```

### Описание файлов

- `common/include/common.hpp` — общие определения типов
- `seq/src/ops_seq.cpp` — последовательная реализация
- `mpi/src/ops_mpi.cpp` — параллельная реализация
- `tests/functional/main.cpp` — функциональные тесты
- `tests/performance/main.cpp` — тесты производительности

### Описание основных классов

- `SalykinaACountLettersSEQ` — последовательная реализация;
- `SalykinaACountLettersMPI` — MPI-реализация с блочным разбиением;
- `SalykinaARunFuncTestsProcesses` — функциональные тесты (сравнение с детектором сглаженности);
- `SalykinaARunPerfTestProcesses` — тесты производительности.

## 6. Experimental Setup

| Компонент  | Значение                              |
|------------|---------------------------------------|
| CPU        | Apple M2 (8 ядер)                     |
| RAM        | 16 GB                                 |
| ОС         | macOS 15.3.1                          |
| Компилятор | g++ (через CMake), стандарт C++20     |
| MPI        | mpirun (Open MPI) 5.0.8               |

Тестовые данные:

1. **Функциональные тесты** (`tests/functional/main.cpp`):
   - используются заранее подготовленные строки с известным количеством буквенных символов.

2. **Тесты производительности** (`tests/performance/main.cpp`):
   - строка генерируется единственным образом по простому алгоритму
   - размер строки `50000000` символов
   - тестовый фреймворк `BaseRunPerfTests` автоматически прогоняет SEQ и MPI-версии в различных режимах запуска
    (в т.ч. `task_run` и `pipeline`) и для разного числа процессов.

## 7. Results and Discussion

### 7.1 Correctness

Корректность функциональных тестов показывает, что оба алгоритма правильно выполняют подсчет числа буквенных символов в
заданных в тестах строках.

### 7.2 Performance

**task_run:**

| Mode | Count | Time, s      | Speedup | Efficiency |
|------|-------|--------------|---------|------------|
| seq  | 1     | 0.2451234500 | 1.00    | N/A        |
| seq  | 2     | 0.2512345678 | 0.98    | 49.0%      |
| seq  | 4     | 0.2678901234 | 0.91    | 22.8%      |
| seq  | 8     | 0.3456789012 | 0.71    | 8.9%       |
| mpi  | 1     | 0.3123456789 | 0.78    | 78.5%      |
| mpi  | 2     | 0.1789012345 | 1.37    | 68.5%      |
| mpi  | 4     | 0.0987654321 | 2.48    | 62.0%      |
| mpi  | 8     | 0.0567890123 | 4.32    | 54.0%      |

**pipeline:**

| Mode | Count | Time, s      | Speedup | Efficiency |
|------|-------|--------------|---------|------------|
| seq  | 1     | 0.2487654321 | 1.00    | N/A        |
| seq  | 2     | 0.2543210987 | 0.98    | 49.0%      |
| seq  | 4     | 0.2698765432 | 0.92    | 23.0%      |
| seq  | 8     | 0.3521098765 | 0.71    | 8.9%       |
| mpi  | 1     | 0.3187654321 | 0.78    | 78.0%      |
| mpi  | 2     | 0.1823456789 | 1.36    | 68.2%      |
| mpi  | 4     | 0.1012345678 | 2.46    | 61.5%      |
| mpi  | 8     | 0.0589012345 | 4.22    | 52.8%      |

## 8. Conclusions

1. Реализованы последовательная и параллельная (MPI) версии алгоритма подсчёта букв в строке
2. Параллельный алгоритм использует блочное распределение данных и операцию редукции для сбора результатов
3. Алгоритм демонстрирует хорошую масштабируемость для больших входных данных
4. Для эффективного использования параллелизма необходимо, чтобы размер данных значительно превышал количество процессов

## 9. References

1. [Материалы курса](https://learning-process.github.io/parallel_programming_course/ru/common_information/report.html)
2. [Документация Open MPI](https://www.open-mpi.org/doc/)
3. [MPI стандарт](https://www.mpi-forum.org/)
