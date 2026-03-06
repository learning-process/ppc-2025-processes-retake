# Нахождение максимумов столбцов матрицы

- Студент: Черемхин Андрей Александрович, группа 3823Б1ПР3
- Технология: SEQ | MPI
- Вариант: 16

## Введение

Цель задачи — разработать и реализовать последовательную (SEQ) и параллельную (MPI) версии алгоритма нахождения максимумов по каждому столбцу матрицы целых чисел, а также провести проверку корректности и измерение производительности.

## Постановка задачи

На вход подаётся матрица целых чисел. Требуется вычислить вектор, где элемент i равен максимуму элементов i-го столбца.

Тип входных данных:

```cpp
using InType = std::vector<std::vector<int>>;
```

Тип выходных данных:

```cpp
using OutType = std::vector<int>;
```

## Базовый алгоритм (Sequential)

Алгоритм последовательно проходит по столбцам и для каждого столбца вычисляет максимум по всем строкам.

```cpp
std::vector<int> max_in_col(C);
for (int j = 0; j < C; ++j) {
  max_in_col[j] = A[0][j];
  for (int i = 1; i < R; ++i) {
    if (A[i][j] > max_in_col[j]) {
      max_in_col[j] = A[i][j];
    }
  }
}
```

Сложность:

- (O(R*C)) по времени
- (O(C)) по памяти

## Схема распараллеливания (MPI)

Распараллеливание выполняется по столбцам: каждый процесс получает диапазон столбцов и вычисляет максимумы только для них. Далее результаты собираются на корневом процессе и рассылаются всем процессам.

### Инициализация MPI

```cpp
int rank = 0;
int size = 0;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
```

### Расчёт распределения столбцов

Для C столбцов и P процессов:

- `cols_per_process = C / P`
- `remainder = C % P`
- первые `remainder` процессов получают на 1 столбец больше

```cpp
const int cols_per_process = num_cols / size;
const int remainder = num_cols % size;

int start_col = rank * cols_per_process + std::min(rank, remainder);
int cols_for_rank = cols_per_process + (rank < remainder ? 1 : 0);
int end_col = start_col + cols_for_rank;
```

### Вычисление локальных максимумов

Каждый процесс проходит по всем строкам, но только по своим столбцам:

```cpp
std::vector<int> local_max;
for (int j = start_col; j < end_col; ++j) {
  int mx = A[0][j];
  for (int i = 1; i < R; ++i) {
    mx = std::max(mx, A[i][j]);
  }
  local_max.push_back(mx);
}
```

### Сбор результатов на rank 0

- Rank 0 заполняет результат для своего диапазона.
- Затем принимает от остальных процессов их массивы максимумов.

```cpp
if (rank == 0) {
  // MPI_Recv(...) для каждого rank != 0
} else {
  MPI_Send(local_max.data(), (int)local_max.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
}
```

### Рассылка результата всем процессам

После сборки на rank 0 выполняется `MPI_Bcast`, чтобы `GetOutput()` был валиден на каждом процессе:

```cpp
MPI_Bcast(result.data(), num_cols, MPI_INT, 0, MPI_COMM_WORLD);
```

## Детали реализации

Структура задачи:

```
tasks/cheremkhin_a_matr_max_colum/
├── common
│   └── include
│       └── common.hpp
├── info.json
├── mpi
│   ├── include
│   │   └── ops_mpi.hpp
│   └── src
│       └── ops_mpi.cpp
├── report.md
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

Ключевые классы:

- `CheremkhinAMatrMaxColumSEQ` — последовательная реализация
- `CheremkhinAMatrMaxColumMPI` — MPI реализация

## Экспериментальная среда

| Компонент | Значение |
|---|---|
| ОС | Linux 6.6.87.2-microsoft-standard-WSL2 |
| Компилятор | g++ 13.3.0 |
| CMake | 3.28.3 |
| MPI | Open MPI 4.1.6 |

Сборка (пример):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Переменные окружения тест-раннера:

- `PPC_NUM_THREADS`, `PPC_NUM_PROC`, `PPC_ASAN_RUN`

Запуск тестов (пример):

```bash
export PPC_NUM_THREADS=4
export PPC_NUM_PROC=4
export PPC_ASAN_RUN=1
python3 scripts/run_tests.py --running-type threads --counts 1 2 4
python3 scripts/run_tests.py --running-type processes --counts 2 4
python3 scripts/run_tests.py --running-type performance
```

## Результаты и обсуждение

### Корректность

Корректность проверяется функциональными тестами `tests/functional/main.cpp`:

- сравнение результата SEQ и MPI с заранее заданным эталоном для нескольких матриц, включая отрицательные значения.

### Производительность

В performance-тесте `tests/performance/main.cpp` генерируется детерминированная матрица 1024×1024, для которой заранее вычисляется эталонный вектор максимумов.

Команды (пример):

```bash
./build/bin/ppc_perf_tests --gtest_color=0 --gtest_filter='*_seq_*'

ASAN_OPTIONS=detect_leaks=0 mpirun --allow-run-as-root -x ASAN_OPTIONS -np 2 \
  ./build/bin/ppc_perf_tests --gtest_color=0 --gtest_filter='*_mpi_*'
```

Ниже — результаты одного локального прогона (в секундах).

task_run (базовая точка: SEQ, 1 процесс):

| Mode | Count | Time, s | Speedup | Efficiency |
|------|------:|--------:|--------:|-----------:|
| seq  | 1 | 0.0192850 | 1.00 | N/A |
| mpi  | 2 | 0.0038811 | 4.97 | 248.5% |
| mpi  | 4 | 0.0033862 | 5.69 | 142.4% |

pipeline (базовая точка: SEQ, 1 процесс):

| Mode | Count | Time, s | Speedup | Efficiency |
|------|------:|--------:|--------:|-----------:|
| seq  | 1 | 0.0197566 | 1.00 | N/A |
| mpi  | 2 | 0.0045862 | 4.31 | 215.4% |
| mpi  | 4 | 0.0028570 | 6.92 | 172.9% |

Примечания:

- Значения speedup/efficiency могут быть выше 100% из-за особенностей измерений (короткие задачи, влияние кэшей/планировщика, различия в запуске под `mpirun` и без него). Для более устойчивых результатов стоит:
  - увеличивать число прогонов (`num_running`)
  - измерять без санитайзеров и с закреплением потоков/процессов (pinning)

## Заключение

Разработаны SEQ и MPI реализации вычисления максимумов по столбцам матрицы.

- SEQ-версия имеет сложность (O(R*C)) и является базовой.
- MPI-версия распараллеливает работу по столбцам и использует обмены `MPI_Send/MPI_Recv` + `MPI_Bcast`, чтобы результат был доступен на всех процессах.
- Реализованы функциональные и performance тесты, позволяющие проверять корректность и измерять время выполнения.

## Источники

1. [Материалы курса: отчёт (требования и структура)](https://learning-process.github.io/parallel_programming_course/ru/common_information/report.html#overview-and-placement)

