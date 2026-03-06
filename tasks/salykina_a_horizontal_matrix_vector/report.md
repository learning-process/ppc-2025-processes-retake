# Ленточная горизонтальная схема - умножение матрицы на вектор

- Студент: Салыкина Алёна Игоревна, 3823Б1ПР3
- Технология: SEQ | MPI
- Вариант: 11

## 1. Introduction

Операция умножени матрицы на вектор — фундаментальная операция линейной алгебры, широко используемая в вычислительной
математике, машинном обучении и обработке сигналов.

Когда приходится работать с большими матрицами, размерности которых достигают тысяч, последовательный
метод оказывается малоэффективным. В связи с этим возникает потребность в создании параллельного алгоритма.

## 2. Problem Statement

Разработать:

1. Последовательный алгоритм: Вычислить $y$ = ($A \times x$), где $A$ — ленточная матрица с горизонтальной
схемой хранения (строки как блоки).
2. Параллельный алгоритм: Распределить строки матрицы по $k$ процессорам; каждый вычисляет свою часть $y_i$,
синхронизируя доступ к $x$

Входные данные:

- Матрица $A$ размерности ($m \times n$)
- Вектор $x$ размерности $n$

Выходные данные:

- Вектор $y$ размерности $m$

## 3. Baseline Algorithm (Sequential)

Последовательный алгоритм выполняет прямое вычисление скалярного произведения для каждой строки матрицы $A$.

Внешний цикл итерируется по строкам, внутренний цикл вычисляет скалярное произведение строки $i$ на вектор $x$.

```cpp

 for (int i = 0; i < rows; ++i) {
  double temp = 0.0;
  for (int j = 0; j < cols; ++j) {
    const std::size_t idx =
        (static_cast<std::size_t>(i) * static_cast<std::size_t>(cols)) + static_cast<std::size_t>(j);
    temp += matrix[idx] * vec[static_cast<std::size_t>(j)];
  }
  res[static_cast<std::size_t>(i)] = temp;
}

```

## 4. Parallelization Scheme

Для распараллеливания используется ленточная горизонтальная схема, при которой строки матрицы $A$ распределяются
по блокам между $k$ процессами.

Входная матрица $A$ делится на $k$ блоков строк. Распределение осуществляется корневым процессом (Rank 0)
с помощью `MPI_Scatterv`, что позволяет обрабатывать блоки переменной длины в
зависимости от остатка ($M \bmod k$) для обеспечения равномерной нагрузки.

Вычисление каждого элемента $y_i$ является независимым и требует доступа только к соответствующей строке $A_i$ и
полному вектору $x$. Обмен граничными данными не требуется.

1. Получение текущего ранга и количетсво процессов

```cpp
int rank = 0;
int size = 0;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
```

2. Чтение входных данных на root-процессе

```cpp
int rows = 0;
int cols = 0;
std::vector<double> vec;

if (rank == 0) {
  rows = std::get<1>(GetInput());
  cols = std::get<2>(GetInput());
  vec = std::get<3>(GetInput());
}
```

3. Рассылка размерностей и вектора

```cpp
MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

if (rank != 0) {
  vec.resize(static_cast<size_t>(cols));
}
if (cols > 0) {
  MPI_Bcast(vec.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
```

4. Расчет распределения строк

```cpp
std::vector<int> rows_counts;
std::vector<int> rows_displs;
CalculateDistribution(rows, size, rows_counts, rows_displs);
```

5. Подготовка параметров для Scatterv

```cpp
std::vector<int> send_counts(static_cast<size_t>(size));
std::vector<int> send_displs(static_cast<size_t>(size));
for (int i = 0; i < size; ++i) {
  send_counts[static_cast<size_t>(i)] = rows_counts[static_cast<size_t>(i)] * cols;
  send_displs[static_cast<size_t>(i)] = rows_displs[static_cast<size_t>(i)] * cols;
}

int my_rows = rows_counts[static_cast<size_t>(rank)];
int my_data_size = my_rows * cols;
```

6. Выделение памяти и рассылка блоков матрицы процессам

```cpp
std::vector<double> local_matrix;
if (my_data_size > 0) {
  local_matrix.resize(static_cast<size_t>(my_data_size));
}

const double *sendbuf = nullptr;
if (rank == 0) {
  sendbuf = std::get<0>(GetInput()).data();
}

MPI_Scatterv(sendbuf, send_counts.data(), send_displs.data(), MPI_DOUBLE,
             (my_data_size > 0) ? local_matrix.data() : nullptr, my_data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

7. Локальное умножение матрицы на вектор

```cpp
std::vector<double> local_res;
if (my_rows > 0) {
  local_res.resize(static_cast<size_t>(my_rows));
}

for (int i = 0; i < my_rows; ++i) {
  double sum = 0.0;
  for (int j = 0; j < cols; ++j) {
    const std::size_t idx = (static_cast<std::size_t>(i) * static_cast<std::size_t>(cols)) + static_cast<std::size_t>(j);
    sum += local_matrix[idx] * vec[static_cast<std::size_t>(j)];
  }
  local_res[static_cast<size_t>(i)] = sum;
}
```

8. Сбор результатов

```cpp
MPI_Allgatherv((my_rows > 0) ? local_res.data() : nullptr, my_rows, MPI_DOUBLE, GetOutput().data(),
               rows_counts.data(), rows_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
```

## 5. Implementation Details

### Структура каталога задачи

```text
tasks/salykina_a_horizontal_matrix_vector/
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

### Вспомогательные функции

- Вспомогательная функция `CalculateDistribution` вычисляет массивы `counts` и `displs` для
 обеспечения равномерного распределения строк матрицы и соответствующих элементов вектора между процессами.

## 6. Experimental Setup

| Компонент  | Значение                              |
|------------|---------------------------------------|
| CPU        | Apple M2 (8 ядер)                     |
| RAM        | 16 GB                                 |
| ОС         | macOS 15.3.1                          |
| Компилятор | g++ (через CMake), стандарт C++20     |
| MPI        | mpirun (Open MPI) 5.0.8               |

Тестовые данные:

1. Набор функциональных тестов (kTestParam) включает в себя 8 сценариев, охватывающих различные
краевые и типовые случаи, включая квадратные, прямоугольные,
и минимальные размеры: $(6, 6), (4, 2), (2, 4), (1, 1), (2, 1), (1, 2), (2, 2), (3, 7)$.

2. Тесты производительности проводятся на фиксированном размере матрицы $\mathbf{N=8000}$ и $\mathbf{M=8000}$.
Вектор $X$ заполняется константным значением 0.5, а матрица $A$ заполняется случайными значениями.

Генерация данных:

- Вектор $x$ заполняется константным значением $\mathbf{1.0}$.

- Матрица $A$ заполняется по формуле $A_{ij} = i + j$.

Ожидаемый результат:

- Вектор $y$ вычисляется как сумма элементов соответствующей строки $A$, поскольку $X_j = 1$.

Проверка корректности:

- Используется абсолютный допуск $\mathbf{1e-5}$ для сравнения результатов с плавающей точкой между последовательной
и параллельной версиями.

## 7. Results and Discussion

### 7.1 Correctness

Корректность проверялась с использованием набора функциональных тестов, включающих малые матрицы и большие матрицы
 для проверки краевых условий. Проверена точность вычислений с плавающей точкой.

Результаты параллельной версии (MPI) полностью совпали с результатами последовательной (SEQ) реализации
 для всех тестовых случаев.

### 7.2 Performance

|Режим|Число процессов|Время(мс)|Ускорение|Эффективность|
|-----|---------------|---------|---------|-|
|seq|1|645|1||
|mpi|4|884|0.73|18.3%|
|mpi|8|1158|0.56|7%|
|mpi|12|1566|0.41|3.4%|
|mpi|16|3207|0.20|1.3%|

Наблюдается анти-ускорение (деградация производительности): время выполнения монотонно растет с увеличением числа процессов,
 и, начиная с $P=4$, ускорение $S_p$ становится меньше 1. Это означает, что параллельная версия работает медленнее,
чем последовательная. В теории это связано с слишком большими расходами на mpi реализацию и отсутствием у процессора
стольки ядер, что и влечёт такую производительность.

## 8. Conclusions

Была разработана параллельная реализация задачи умножения матрицы на вектор с использованием MPI и горизонтальной ленточной
схемы.Вопреки ожиданиям для задачи с высокой интенсивностью вычислений, тестирование показало сильную деградацию производительности.

## 9. References

1. [Материалы курса](https://learning-process.github.io/parallel_programming_course/ru/common_information/report.html)
2. [Документация Open MPI](https://www.open-mpi.org/doc/)
3. [MPI стандарт](https://www.mpi-forum.org/)
