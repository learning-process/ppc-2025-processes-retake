# Топологии сетей передачи данных. Гиперкуб

* Студент: Дергунов Сергей Антонович
* Группа: 3823Б1ПР4
* Вариант: 10
* Технологии: SEQ, MPI

---

## 1. Введение

Параллельное программирование широко используется при разработке приложений, требующих высокой производительности и интенсивного обмена данными между процессами. 
В данной работе рассматривается практическая реализация передачи сообщений в среде MPI с использованием топологии гиперкуба.

Цель данной работы - реализовать передачу данных между процессами по топологии гиперкуба без использования встроенных функций MPI для создания топологий (`MPI_Cart_create`, `MPI_Graph_create`), используя только базовые операции обмена (`MPI_Send`, `MPI_Recv`).

Основные задачи:
1. Реализовать вычисление пути передачи данных между произвольными процессами.
2. Смоделировать передачу данных по гиперкубу.
3. Оценить корректность работы и производительность MPI-версии.

---

## 2. Постановка задачи

### Входные данные (InType):
```cpp
struct HypercubeMessage {
  int source;     // процесс-отправитель
  int dest;       // процесс-получатель
  int data_size;  // количество элементов для генерации и передачи
};

### Выходные данные (OutType):
* data — сумма элементов вектора, полученного процессом-получателем.

### Ограничения:
* Количество процессов world_size может не быть степенью двойки.
* Процессы source и dest должны быть в диапазоне 0 <= source < world_size.

Суммирование элементов используется как простая вычислительная операция, позволяющая проверить корректность передачи данных и целостность полученного сообщения.

---

## 3. Базовый алгоритм (вычисление пути)

### 3.1. Структура гиперкуба

Для 3-мерного гиперкуба:

      110───────111
      ╱│        ╱│
     ╱ │       ╱ │
   010─┼─────011 │
    │ 100─────│─101
    │ ╱       │ ╱
   000───────001

Каждый процесс соединён с соседями, отличающимися на один бит.

### 3.2. Алгоритм передачи

1. Определение размерности гиперкуба по числу процессов (CalculateHypercubeDimension).
2. Вычисление кратчайшего пути между source и dest с помощью XOR-битовой операции (CalcPath).
3. Последовательная передача данных по пути (MPI_Send/MPI_Recv).
4. Процесс-получатель рассылает результат всем процессам (MPI_Bcast).

Такой подход позволяет на каждом шаге изменять ровно один бит номера процесса, что соответствует переходу к соседу в гиперкубе и гарантирует минимальную длину маршрута.

Пример функции расчёта пути:

std::vector<int> DergynovSHypercube::CalcPath(int source, int dest, int dimensions) {
    std::vector<int> path;
    int current = source;
    path.push_back(current);
    int xor_val = source ^ dest;
    for (int dim = 0; dim < dimensions; dim++) {
        int mask = 1 << dim;
        if ((xor_val & mask) != 0) {
            current ^= mask;
            path.push_back(current);
            if (current == dest) break;
        }
    }
    return path;
}

---

## 4. Схема распараллеливания

Роли процессов:
* Source — генерирует данные и инициирует передачу.
* Промежуточные узлы — пересылают данные следующему процессу.
* Dest — получает данные, суммирует элементы, рассылает всем.
* Остальные — получают результат через MPI_Bcast.

Коммуникация:
* Каждому процессу известен свой ранг и количество процессов.
* Передача строго по рассчитанному пути.
* Используются только MPI_Send и MPI_Recv.

Описанная схема напрямую отражена в методе RunImpl, где для каждого процесса определяется его позиция в маршруте и соответствующая роль в передаче данных.

bool DergynovSHypercubeMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &in = GetInput();
  int src = in[0];
  int dst = in[1];
  int data_size = in[2];

  std::vector<int> data;
  if (rank == src) {
    data.resize(data_size, 1);
  }

  if (src == dst) {
    MPI_Bcast(&data_size, 1, MPI_INT, dst, MPI_COMM_WORLD);
    if (rank != dst) {
      data.resize(data_size);
    }
    MPI_Bcast(data.data(), data_size, MPI_INT, dst, MPI_COMM_WORLD);
    GetOutput() = std::accumulate(data.begin(), data.end(), 0);
    return true;
  }

  int dim = CalcDim(size);
  auto path = BuildPath(src, dst, dim);

  int pos, next, prev;
  FindPos(rank, path, pos, next, prev);

  if (pos != -1) {
    if (rank == src) {
      BusyWork(120000);
      SendVec(data, next);
    } else if (rank == dst) {
      RecvVec(data, prev);
      BusyWork(120000);
    } else {
      RecvVec(data, prev);
      BusyWork(120000);
      SendVec(data, next);
    }
  }

  int final_size = static_cast<int>(data.size());
  MPI_Bcast(&final_size, 1, MPI_INT, dst, MPI_COMM_WORLD);
  if (pos == -1) {
    data.resize(final_size);
  }
  MPI_Bcast(data.data(), final_size, MPI_INT, dst, MPI_COMM_WORLD);

  GetOutput() = std::accumulate(data.begin(), data.end(), 0);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

---

## 5. Реализация

bool DergynovSHypercube::RunImpl() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto input = GetInput();
    int source = input[0], dest = input[1];
    std::vector<int> data(input[2], 1);

    int dimensions = CalculateHypercubeDimension(size);
    std::vector<int> path = CalcPath(source, dest, dimensions);

    int my_pos = -1, prev_neighbor = -1, next_neighbor = -1;
    CalcPositions(rank, path, my_pos, next_neighbor, prev_neighbor);

    if (my_pos != -1) {
        if (rank == source) SendData(data, next_neighbor);
        else if (rank == dest) { RecvData(data, prev_neighbor); PerformComputeLoad(150000); }
        else { RecvData(data, prev_neighbor); PerformComputeLoad(150000); SendData(data, next_neighbor); }
    }

    MPI_Bcast(data.data(), data.size(), MPI_INT, dest, MPI_COMM_WORLD);
    GetOutput() = std::accumulate(data.begin(), data.end(), 0);
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
}

В приведённом фрагменте показана основная логика работы MPI-версии: определение маршрута, приём и передача данных между соседними узлами, а также финальная рассылка результата всем процессам.
Вспомогательные функции инкапсулируют детали обмена и вычислительной нагрузки.
Для каждого узла выполняется PerformComputeLoad(150000), чтобы оценить влияние вычислений на время передачи.

---

## 6. Экспериментальная часть

### Аппаратная конфигурация

* Docker-контейнер (Linux)
* GCC
* OpenMPI
* Режим сборки: Release

### Команды запуска

mpirun --allow-run-as-root -n 1 ./build/bin/ppc_perf_tests
mpirun --allow-run-as-root -n 2 ./build/bin/ppc_perf_tests
mpirun --allow-run-as-root -n 4 ./build/bin/ppc_perf_tests

---

## 7. Результаты

### 7.1 Корректность

Функциональные тесты пройдены для всех сценариев:
* соседние и удалённые процессы
* передача самому себе
* пустой массив
* обратная передача

Все процессы получают корректную сумму элементов.

### 7.2. Производительность

| Процессы | Время выполнения (с) | Шаги передачи |
| -------- | -------------------- | ------------- |
| 2        | 0.0015               | 1             |
| 4        | 0.0016               | 2             |
| 8        | 0.0017               | 3             |
| 16       | 0.0017               | 4             |
| 32       | 0.0023               | 5             |

Комментарий:
* Время растёт с числом процессов из-за увеличения длины пути.
* Основное время тратится на имитацию вычислительной нагрузки.
* MPI-версия эффективна, накладные расходы минимальны.

Полученные результаты соответствуют теоретическим ожиданиям для гиперкуба: при увеличении числа процессов растёт размерность топологии и, как следствие, число шагов передачи. Это подтверждает корректность вычисления маршрута.

---

## 8. Заключение

* Реализована корректная передача данных по топологии гиперкуба.
* MPI-версия не использует встроенные функции топологий.
* Функциональные тесты подтверждают корректность работы.
* Производительность остаётся высокой даже при увеличении числа процессов.

Дальнейшие улучшения:
* Увеличение объёма данных для более точной оценки масштабируемости.
* Тестирование на реальном кластере с большим числом узлов.

---

## 9. Литература

1. Gropp W., Lusk E., Skjellum A. Using MPI
2. MPI Standard Documentation
3. Документация GoogleTest
4. Сысоев А. В. Лекции по параллельному программированию