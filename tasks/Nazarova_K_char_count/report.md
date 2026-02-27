# Подсчёт вхождений символа в тексте

- Студент: Назарова К, группа 3823Б1ПР3
- Технология: SEQ | MPI
- Вариант: 23

## Введение

Цель работы — реализовать последовательную (SEQ) и параллельную (MPI) версии алгоритма подсчёта количества вхождений заданного символа в массиве символов, а также проверить корректность и измерить производительность.

## Постановка задачи

Дан текст (массив символов) и целевой символ `target`. Требуется вычислить количество индексов i, для которых `text[i] == target`.

Тип входных данных:

```cpp
struct Input {
  std::vector<char> text;
  char target;
};
```

Тип выходных данных:

```cpp
using OutType = int;
```

Ограничения:

- допускается пустой текст n = 0;
- значение `target` может быть любым, включая 0.

## Базовый алгоритм (Sequential)

Последовательная версия последовательно проходит по всем элементам массива и увеличивает счётчик, если текущий символ равен `target`:

```cpp
int cnt = 0;
for (char c : text) {
  if (c == target) ++cnt;
}
```

Сложность:

- O(n) по времени;
- O(1) по дополнительной памяти (кроме хранения входа).

## Схема распараллеливания (MPI)

Распараллеливание выполняется по элементам текста: каждый процесс обрабатывает свой непрерывный диапазон индексов.

Далее:

- каждый процесс вычисляет `local_count` на своём фрагменте;
- итоговый ответ получается коллективной операцией `MPI_Allreduce(local_count, SUM)`, чтобы результат был доступен на каждом процессе.

Коммуникация: один коллектив (`MPI_Allreduce`) после локального вычисления.

## Детали реализации

Структура задачи:

```
tasks/Nazarova_K_char_count/
├── common
│   └── include
│       └── common.hpp
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
└── tests
    ├── functional
    │   └── main.cpp
    └── performance
        └── main.cpp
```

Ключевые моменты реализации:

- Общие типы: `tasks/Nazarova_K_char_count/common/include/common.hpp`
  - `Input { std::vector<char> text; char target; }`
  - `OutType = int`
- SEQ-версия: `tasks/Nazarova_K_char_count/seq/src/ops_seq.cpp`
  - подсчёт выполнен через `std::count(text.begin(), text.end(), target)`.
- MPI-версия: `tasks/Nazarova_K_char_count/mpi/src/ops_mpi.cpp`
  - локальный подсчёт через `std::count` на своём диапазоне;
  - объединение результата через `MPI_Allreduce`.

Пограничные случаи:

- пустой текст: каждый процесс получает `0`;
- при p > n часть процессов обрабатывает пустые диапазоны, что корректно учитывается в `MPI_Allreduce`.

## Экспериментальная среда

| Компонент | Значение |
|---|---|
| ОС | Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2) |
| CPU | 12th Gen Intel(R) Core(TM) i5-1235U, 6 cores / 12 threads (x86_64) |
| CMake | сборка `Release` |
| Компилятор | gcc-14 / g++-14 (из `build/CMakeCache.txt`) |
| MPI | Open MPI 4.1.6 (`mpirun --version`) |

Переменные окружения для локальных запусков:

- `PPC_NUM_THREADS=4`
- `PPC_NUM_PROC=2` (для MPI)
- `PPC_ASAN_RUN=1`

Примечание: в devcontainer MPI запускается под `root`, поэтому для OpenMPI требуется `--allow-run-as-root` или переменные `OMPI_ALLOW_RUN_AS_ROOT=1` и `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`.

## Как воспроизвести

Сборка из корня репозитория:

```bash
git submodule update --init --recursive

cmake -S . -B build -D USE_FUNC_TESTS=ON -D USE_PERF_TESTS=ON -D CMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

Запуск тестов через скрипт курса:

```bash
export PPC_NUM_THREADS=4
export PPC_NUM_PROC=2
export PPC_ASAN_RUN=1

python3 scripts/run_tests.py --running-type threads --counts 1 2 4
python3 scripts/run_tests.py --running-type processes --counts 2 4
python3 scripts/run_tests.py --running-type performance
```

Дополнительно: прямой запуск MPI-тестов именно для этой задачи:

```bash
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export PPC_NUM_THREADS=4
export OMP_NUM_THREADS=4

mpirun -np 2 -x PPC_NUM_THREADS -x OMP_NUM_THREADS ./build/bin/ppc_func_tests \
  --gtest_filter='*nazarova_k_char_count_processes_mpi_enabled*'

./build/bin/ppc_perf_tests --gtest_filter='*nazarova_k_char_count_processes_seq_enabled*'
mpirun -np 2 -x PPC_NUM_THREADS -x OMP_NUM_THREADS ./build/bin/ppc_perf_tests \
  --gtest_filter='*nazarova_k_char_count_processes_mpi_enabled*'
```

## Результаты и обсуждение

### Корректность

Корректность проверяется функциональными тестами в `tasks/Nazarova_K_char_count/tests/functional/main.cpp`.

В тестах проверяются случаи:

- пустой текст;
- отсутствие вхождений `target`;
- детерминированная расстановка `target` в тексте для проверки ожидаемого количества.

Корректность MPI-версии дополнительно подтверждается запуском под `mpirun -np 2`.

### Производительность

В performance-тесте создаётся вектор длины 10^6, выбирается `target='a'`, и целевой символ размещается на каждом 7-м индексе. Результат — суммарное число совпадений.

Времена (режим Perf `task_run`):

| Mode | Count |  Time, s  | Speedup | Efficiency |
|------|------:|----------:|--------:|-----------:|
| seq  |   1   | 0.0002117 |   1.00  | N/A        |
| mpi  |   2   | 0.0001194 |   1.77  | 88.7%      |

Замечания:

- операция очень лёгкая (сравнение символов), поэтому накладные расходы MPI заметны;
- при увеличении числа процессов ускорение может ограничиваться стоимостью `MPI_Allreduce` и пропускной способностью памяти.

## Заключение

Реализованы SEQ и MPI версии подсчёта вхождений символа в тексте.

- SEQ-алгоритм имеет сложность O(n) и служит базовой реализацией.
- MPI-версия делит вход по индексам и использует `MPI_Allreduce`, обеспечивая корректный итог на каждом процессе.
- На тестовом размере 10^6 при 2 процессах получено ускорение 1.77.
