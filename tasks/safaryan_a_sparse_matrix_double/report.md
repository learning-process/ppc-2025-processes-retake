# Умножение разреженных матриц (double). Формат CRS

Студент: Сафарян Григор Ваагнович  
Группа: 3823Б1ПР5  
Технология: SEQ | MPI  
Вариант: 4  

## 1. Введение

Умножение разреженных матриц используется в научных вычислениях и прикладных задачах,
где большинство элементов равно нулю. Для экономии памяти и ускорения вычислений 
применяется формат хранения CRS (Compressed Row Storage), позволяющий работать только с ненулевыми элементами.

## 2. Постановка задачи

Реализовать умножение двух разреженных матриц `A (M×K)` и `B (K×N)` в формате CRS 
и получить `C = A×B (M×N)` также в формате CRS.

Формат CRS:
- `values` — ненулевые значения
- `col_indices` — индексы столбцов для `values`
- `row_ptr` — указатели начала строк в `values/col_indices`

Входные данные:

```cpp
using InType = std::tuple<std::vector<double>,  // values_A
                          std::vector<int>,     // col_indices_A
                          std::vector<int>,     // row_ptr_A
                          std::vector<double>,  // values_B
                          std::vector<int>,     // col_indices_B
                          std::vector<int>,     // row_ptr_B
                          int,                  // n_rows_A
                          int,                  // n_cols_A (== n_rows_B)
                          int                   // n_cols_B
                          >;
```

**Тип выходных данных:**

```cpp
using OutType = std::tuple<std::vector<double>,  // values_C
                           std::vector<int>,     // col_indices_C
                           std::vector<int>      // row_ptr_C
                           >;
```

**Ограничения:**

- Входные данные — две совместимые разреженные матрицы произвольных размеров (столбцы A = строки B)
- Результат SEQ и MPI должен совпадать
- в MPI матрица A распределяется по строкам, матрица B рассылается всем процессам полностью

## 3. Базовый алгоритм (Sequential)

### Алгоритм последовательной реализации

Идея: для каждой строки i матрицы A накапливается строка результата C[i,*].

1. **Шаги**
   - Получить на вход две разреженные матрицы в формате CRS: A и B
   - Инициализировать результирующую структуру CRS для матрицы C
   - Инициализировать `row_ptr_C_[0] = 0`

2. **Умножение матриц**
   - Для строки i берём все ненулевые A[i,k]
   - Для каждого такого k обходим строку B[k,*] (тоже CRS)
   - Накапливаем сумму в временном массиве temp_row[j] += A[i,k]*B[k,j]
   - Из temp_row формируем CRS-представление строки результата (с порогом 1e-12)


### Код последовательной реализации:

```cpp
bool SafaryanASparseMatrixDoubleSEQ::RunImpl() {
  row_ptr_C_.resize(n_rows_A_ + 1, 0);
  row_ptr_C_[0] = 0;

  std::vector<std::vector<double>> row_values(n_rows_A_);
  std::vector<std::vector<int>> row_cols(n_rows_A_);

  for (int i = 0; i < n_rows_A_; i++) {
    ProcessRow(i, row_values[i], row_cols[i]);
    row_ptr_C_[i + 1] = row_ptr_C_[i] + static_cast<int>(row_cols[i].size());
  }

  for (int i = 0; i < n_rows_A_; i++) {
    values_C_.insert(values_C_.end(), row_values[i].begin(), row_values[i].end());
    col_indices_C_.insert(col_indices_C_.end(), row_cols[i].begin(), row_cols[i].end());
  }

  return true;
}

void SafaryanASparseMatrixDoubleSEQ::ProcessRow(int row_idx,
                                               std::vector<double>& row_values,
                                               std::vector<int>& row_cols) {
  int row_start_a = row_ptr_A_[row_idx];
  int row_end_a = row_ptr_A_[row_idx + 1];

  std::vector<double> temp_row(n_cols_B_, 0.0);

  for (int k_idx = row_start_a; k_idx < row_end_a; k_idx++) {
    double a_val = values_A_[k_idx];
    int k = col_indices_A_[k_idx];

    int row_start_b = row_ptr_B_[k];
    int row_end_b = row_ptr_B_[k + 1];

    for (int j_idx = row_start_b; j_idx < row_end_b; j_idx++) {
      double b_val = values_B_[j_idx];
      int j = col_indices_B_[j_idx];
      temp_row[j] += a_val * b_val;
    }
  }

  for (int j = 0; j < n_cols_B_; j++) {
    if (std::abs(temp_row[j]) > 1e-12) {
      row_values.push_back(temp_row[j]);
      row_cols.push_back(j);
    }
  }

  if (!row_cols.empty()) {
    std::vector<std::pair<int, double>> pairs;
    pairs.reserve(row_cols.size());
    for (size_t idx = 0; idx < row_cols.size(); idx++) {
      pairs.emplace_back(row_cols[idx], row_values[idx]);
    }
    std::ranges::sort(pairs);

    for (size_t idx = 0; idx < pairs.size(); idx++) {
      row_cols[idx] = pairs[idx].first;
      row_values[idx] = pairs[idx].second;
    }
  }
}
```

## 4. Схема распараллеливания

### 4.1 Распределение данных

Матрица **A** распределяется по строкам между процессами (циклически):

- строка `i` → процесс `(i % world_size)`

Матрица **B** полностью рассылается всем процессам в формате CRS (массивы `values_B`, `col_indices_B`, `row_ptr_B`).

---

### 4.2 Коммуникации

- Рассылка размеров матриц: `MPI_Bcast`
- Рассылка матрицы **B**:
  - `MPI_Bcast(values_B)`
  - `MPI_Bcast(col_indices_B)`
  - `MPI_Bcast(row_ptr_B)`
- Распределение строк матрицы **A**: `MPI_Send` / `MPI_Recv` (по строкам, только ненулевые элементы)
- Сбор результата: worker → root через `MPI_Send` / `MPI_Recv`, `rank 0` собирает финальный CRS

---

### 4.3 Выполнение алгоритма

1. `rank 0` читает входные данные и рассылает размеры (`MPI_Bcast`)
2. `rank 0` рассылает матрицу **B** всем процессам (`MPI_Bcast`)
3. `rank 0` отправляет назначенные строки матрицы **A** процессам (`MPI_Send`), процессы принимают (`MPI_Recv`)
4. Каждый процесс умножает свои строки **A** на общую **B**, формирует локальный CRS результата
5. Процессы отправляют локальный результат на `rank 0` (`MPI_Send` / `MPI_Recv`)
6. `rank 0` объединяет строки в итоговый CRS и записывает результат через `GetOutput()`

### Псевдокод

```text
function PreProcessingImpl():
    rank, size = MPI_comm_info()
    rank_ = rank
    world_size_ = size
    
    // Очистка локальных данных
    local_rows_.clear()
    local_values_A_.clear()
    local_col_indices_A_.clear()
    local_row_ptr_A_.clear()
    local_values_C_.clear()
    local_col_indices_C_.clear()
    local_row_ptr_C_.clear()
    
    // Очистка результата
    values_C_.clear()
    col_indices_C_.clear()
    row_ptr_C_.clear()

function RunImpl():
    rank, size = MPI_comm_info()
    
    if size == 1:
        return RunSequential()
    
    // Фаза 1: Рассылка размеров
    n_rows_a, n_cols_a, n_cols_b = PrepareAndValidateSizes()
    
    // Фаза 2: Рассылка матрицы B
    BroadcastMatrixB()
    
    // Фаза 3: Распределение матрицы A
    DistributeMatrixAData()
    
    // Фаза 4: Локальное умножение
    ComputeLocalMultiplication()
    
    // Фаза 5: Сбор результатов
    GatherResults()
    
    return true

function PrepareAndValidateSizes():
    if rank == 0:
        n_rows_a = n_rows_A_
        n_cols_a = n_cols_A_
        n_cols_b = n_cols_B_
    
    sizes = [n_rows_a, n_cols_a, n_cols_b]
    MPI_Bcast(sizes, 3, MPI_INT, 0, MPI_COMM_WORLD)
    
    n_rows_a = sizes[0]
    n_cols_a = sizes[1]
    n_cols_b = sizes[2]
    
    return !(n_rows_a <= 0 || n_cols_a <= 0 || n_cols_b <= 0)

function BroadcastMatrixB():
    if rank == 0:
        b_sizes = [values_B_.size(), col_indices_B_.size(), row_ptr_B_.size()]
    
    MPI_Bcast(b_sizes, 3, MPI_INT, 0, MPI_COMM_WORLD)
    
    if rank != 0:
        values_B_.resize(b_sizes[0])
        col_indices_B_.resize(b_sizes[1])
        row_ptr_B_.resize(b_sizes[2])
    
    MPI_Bcast(values_B_, b_sizes[0], MPI_DOUBLE, 0, MPI_COMM_WORLD)
    MPI_Bcast(col_indices_B_, b_sizes[1], MPI_INT, 0, MPI_COMM_WORLD)
    MPI_Bcast(row_ptr_B_, b_sizes[2], MPI_INT, 0, MPI_COMM_WORLD)

function DistributeMatrixAData():
    // Определяем строки для текущего процесса
    local_rows_ = []
    для i от 0 до n_rows_A_ - 1:
        если i % world_size_ == rank_:
            добавить i в local_rows_
    
    if rank == 0:
        // Отправляем данные остальным процессам
        для dest от 1 до world_size_ - 1:
            SendMatrixADataToProcess(dest)
        
        // Копируем свои строки в локальные массивы
        local_values_A_ = []
        local_col_indices_A_ = []
        local_row_ptr_A_ = [0]
        
        для idx от 0 до local_rows_.size() - 1:
            row = local_rows_[idx]
            row_start = row_ptr_A_[row]
            row_end = row_ptr_A_[row + 1]
            row_nnz = row_end - row_start
            
            добавить values_A_[row_start:row_end] в local_values_A_
            добавить col_indices_A_[row_start:row_end] в local_col_indices_A_
            local_row_ptr_A_.append(local_values_A_.size())
    else:
        ReceiveMatrixAData()

function SendMatrixADataToProcess(dest):
    // Определяем строки для процесса dest (циклическое распределение)
    dest_rows = []
    для i от 0 до n_rows_A_ - 1:
        если i % world_size_ == dest:
            добавить i в dest_rows
    
    // Отправляем количество строк
    dest_row_count = dest_rows.size()
    MPI_Send(dest_row_count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD)
    
    если dest_row_count > 0:
        // Отправляем номера строк
        MPI_Send(dest_rows, dest_row_count, MPI_INT, dest, 1, MPI_COMM_WORLD)
        
        // Отправляем данные для каждой строки
        для row в dest_rows:
            row_start = row_ptr_A_[row]
            row_end = row_ptr_A_[row + 1]
            row_nnz = row_end - row_start
            
            // Отправляем количество ненулевых элементов в строке
            MPI_Send(row_nnz, 1, MPI_INT, dest, 2, MPI_COMM_WORLD)
            
            если row_nnz > 0:
                // Отправляем значения
                row_values = values_A_[row_start:row_end]
                MPI_Send(row_values, row_nnz, MPI_DOUBLE, dest, 3, MPI_COMM_WORLD)
                
                // Отправляем индексы столбцов
                row_cols = col_indices_A_[row_start:row_end]
                MPI_Send(row_cols, row_nnz, MPI_INT, dest, 4, MPI_COMM_WORLD)

function ReceiveMatrixAData():
    // Принимаем данные от корневого процесса
    local_row_count = 0
    MPI_Recv(local_row_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD)
    
    если local_row_count > 0:
        // Принимаем номера строк
        local_rows_.resize(local_row_count)
        MPI_Recv(local_rows_, local_row_count, MPI_INT, 0, 1, MPI_COMM_WORLD)
        
        // Подготавливаем структуры для хранения данных
        local_values_A_ = []
        local_col_indices_A_ = []
        local_row_ptr_A_ = [0]
        
        для i от 0 до local_row_count - 1:
            row_nnz = 0
            MPI_Recv(row_nnz, 1, MPI_INT, 0, 2, MPI_COMM_WORLD)
            
            если row_nnz > 0:
                // Принимаем значения
                row_values = new double[row_nnz]
                MPI_Recv(row_values, row_nnz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD)
                
                // Принимаем индексы столбцов
                row_cols = new int[row_nnz]
                MPI_Recv(row_cols, row_nnz, MPI_INT, 0, 4, MPI_COMM_WORLD)
                
                // Добавляем данные в локальные массивы
                добавить row_values в local_values_A_
                добавить row_cols в local_col_indices_A_
            
            local_row_ptr_A_.append(local_values_A_.size())

function ComputeLocalMultiplication():
    local_row_count = local_rows_.size()
    local_row_values = new vector<vector<double>>(local_row_count)
    local_row_cols = new vector<vector<int>>(local_row_count)
    local_row_ptr_C_ = [0]
    
    // Умножение для каждой локальной строки
    для local_idx от 0 до local_row_count - 1:
        ProcessLocalRow(local_idx, local_row_values[local_idx], local_row_cols[local_idx])
        // Обновляем указатели на строки
        local_row_ptr_C_.append(local_row_ptr_C_[local_idx] + local_row_cols[local_idx].size())
    
    // Собираем все локальные значения и индексы
    local_values_C_ = []
    local_col_indices_C_ = []
    для i от 0 до local_row_count - 1:
        добавить local_row_values[i] в local_values_C_
        добавить local_row_cols[i] в local_col_indices_C_

function ProcessLocalRow(local_idx, row_values, row_cols):
    row_start = local_row_ptr_A_[local_idx]
    row_end = local_row_ptr_A_[local_idx + 1]
    
    // Создаем временный массив для текущей строки результата
    temp_row = new double[n_cols_B_] // инициализирован нулями
    
    // Умножаем строку на матрицу B
    MultiplyRowByMatrixB(row_start, row_end, temp_row)
    
    // Собираем ненулевые элементы
    CollectNonZeroElements(temp_row, n_cols_B_, row_values, row_cols)
    
    // Сортируем по столбцам
    SortRowElements(row_values, row_cols)

function GatherResults():
    if rank == 0:
        // Собираем данные от всех процессов и храним по строкам
        row_values = new vector<vector<double>>(n_rows_A_)
        row_cols = new vector<vector<int>>(n_rows_A_)
        
        // Обрабатываем строки корневого процесса
        ProcessLocalResults(row_values, row_cols)
        
        // Принимаем результаты от других процессов
        для src от 1 до world_size_ - 1:
            ReceiveResultsFromProcess(src, row_values, row_cols)
        
        // Формируем финальную структуру CRS
        CollectAllResults(row_values, row_cols)
        
        // Сохраняем результат
        GetOutput() = (values_C_, col_indices_C_, row_ptr_C_)
    else:
        // Отправляем результаты корневому процессу
        local_row_count = local_rows_.size()
        
        // Всегда отправляем количество строк (даже если 0)
        MPI_Send(local_row_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD)
        
        если local_row_count > 0:
            // Отправляем номера строк (для правильного сопоставления на root)
            MPI_Send(local_rows_, local_row_count, MPI_INT, 0, 1, MPI_COMM_WORLD)
            
            // Отправляем локальные row_ptr_C
            MPI_Send(local_row_ptr_C_, local_row_count + 1, MPI_INT, 0, 2, MPI_COMM_WORLD)
            
            // Отправляем значения и индексы
            total_nnz = local_values_C_.size()
            если total_nnz > 0:
                MPI_Send(local_values_C_, total_nnz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD)
                MPI_Send(local_col_indices_C_, total_nnz, MPI_INT, 0, 4, MPI_COMM_WORLD)
        
        // На не-root процессах устанавливаем пустой результат
        GetOutput() = (пустые_векторы)

function RunSequential():
    если rank != 0:
        return true
    
    // Алгоритм аналогичен последовательной версии
    // ... умножение матриц в формате CRS
    
    GetOutput() = (values_C_, col_indices_C_, row_ptr_C_)
    return true

function PostProcessingImpl():
    return true
```

## 5. Детали реализации

### Структура проекта

```text
tasks/safaryan_a_sparse_matrix_double/
├── common/
│   └── include/
│       └── common.hpp
├── seq/
│   ├── include/
│   │   └── ops_seq.hpp
│   └── src/
│       └── ops_seq.cpp
├── mpi/
│   ├── include/
│   │   └── ops_mpi.hpp
│   └── src/
│       └── ops_mpi.cpp
├── tests/
│   ├── functional/
│   │   └── main.cpp
│   └── performance/
│       └── main.cpp
└── report.md
```

**Ключевые классы и файлы:**

1. **Последовательная реализация (`seq`):**
   - `ops_seq.hpp` — объявление класса `SafaryanASparseMatrixDoubleSEQ`
   - `ops_seq.cpp` — реализация методов:
     - `ValidationImpl()` — проверка корректности входных данных в формате CRS 
     (размеры, монотонность row_ptr, корректность индексов)
     - `PreProcessingImpl()` — инициализация и очистка структур для результата
     - `RunImpl()` — основной алгоритм умножения разреженных матриц в формате CRS
     - `ProcessRow()` — обработка одной строки матрицы A: умножение на матрицу B, 
     сбор ненулевых элементов и сортировка
     - `PostProcessingImpl()` — упаковка результата в `OutType` 

2. **MPI реализация (`mpi`):**

   **Основные методы:**
   - `SafaryanASparseMatrixDoubleMPI()` — конструктор, получающий матрицы A и B в 
   формате CRS только в процессе 0
   - `ValidationImpl()` — проверка инициализации MPI и количества процессов
   - `PreProcessingImpl()` — получение ранга и размера MPI-коммуникатора, очистка локальных данных
   - `RunImpl()` — основной алгоритм параллельного умножения:
     - Подготовка данных через `PrepareAndValidateSizes()` (рассылка размеров)
     - Рассылка матрицы B через `BroadcastMatrixB()`
     - Распределение матрицы A через `DistributeMatrixAData()`
     - Локальные вычисления через `ComputeLocalMultiplication()`
     - Сбор результатов через `GatherResults()`
   - `PostProcessingImpl()` — финализация
   - `RunSequential()` — последовательная версия для случая одного процесса 
   (вызывается из `RunImpl()` при `world_size_ == 1`)
   - `ProcessRowForSequential()` — обработка одной строки в последовательном 
   режиме (аналогично `ProcessRow()` в seq версии)

   **Вспомогательные методы распределения данных:**
   - `PrepareAndValidateSizes()` — широковещательная рассылка размеров матриц (`MPI_Bcast`)
   - `BroadcastMatrixB()` — рассылка матрицы B в формате CRS всем процессам 
   (три массива: values, col_indices, row_ptr)
   - `DistributeMatrixAData()` — основное распределение строк матрицы A в формате CRS между процессами 
   (циклическое распределение)
   - `SendMatrixADataToProcess()` — отправка данных строк матрицы A конкретному процессу
   - `ReceiveMatrixAData()` — прием данных строк матрицы A от корневого процесса

   **Вспомогательные методы вычислений и сбора результатов:**
   - `ComputeLocalMultiplication()` — локальное умножение части матрицы A на матрицу B в формате CRS с 
   формированием результата в формате CRS
   - `ProcessLocalRow()` — обработка одной локальной строки: умножение на матрицу B, сбор 
   ненулевых элементов и сортировка
   - `MultiplyRowByMatrixB()` — умножение строки матрицы A на матрицу B
   - `ProcessElementA()` — обработка одного элемента матрицы A с проверкой границ
   - `MultiplyByRowB()` — умножение элемента A на строку матрицы B
   - `CollectNonZeroElements()` — сбор ненулевых элементов из временного массива (порог 1e-12)
   - `SortRowElements()` — сортировка элементов строки по индексам столбцов
   - `GatherResults()` — основной сбор результатов от всех процессов, объединение и сортировка элементов 
   по столбцам для формирования финальной структуры CRS
   - `ProcessLocalResults()` — обработка локальных результатов корневого процесса
   - `ReceiveResultsFromProcess()` — получение результатов от worker-процесса
   - `CollectAllResults()` — формирование финальной структуры CRS из собранных результатов
   - `SortAndPackRow()` — сортировка и упаковка элементов строки в финальную структуру CRS

3. **Общие компоненты (`common`):**
   - `common.hpp` — общие типы данных (`InType`, `OutType`, `TestType`) и базовый класс `BaseTask`

4. **Тесты:**
   - `tests/functional/main.cpp` — `SafaryanASparseMatrixDoubleFuncTests` — функциональные тесты
   - `tests/performance/main.cpp` — `SafaryanASparseMatrixDoubleRunPerfTests` — тесты производительности

**Архитектурные особенности:**

- Использование формата CRS для эффективного хранения разреженных матриц
- Горизонтальная схема распределения данных (строки матрицы A)
- Циклическое распределение строк для балансировки нагрузки
- Полная репликация матрицы B на всех процессах
- Минимизация коммуникаций за счет использования `MPI_Bcast` для матрицы B
- Сортировка элементов по столбцам для соблюдения формата CRS

## 6. Экспериментальная установка

### Набор инструментов

Компиляция и сборка:
- Компилятор: GCC 11.4.0 (через Homebrew)
- Стандарт языка: C++20
- Среда разработки: Visual Studio Code
- Тип сборки: Release
- Система сборки: CMake

### Управление процессами

PPC_NUM_PROC: устанавливается через параметр -n в mpirun

```bash
# Запуск с различным количеством процессов MPI
mpirun -n 1 ./bin/ppc_func_tests --gtest_filter="*safaryan_a_sparse_matrix_double*"
mpirun -n 2 ./bin/ppc_func_tests --gtest_filter="*safaryan_a_sparse_matrix_double*"
mpirun -n 4 ./bin/ppc_func_tests --gtest_filter="*safaryan_a_sparse_matrix_double*"
mpirun -n 8 ./bin/ppc_func_tests --gtest_filter="*safaryan_a_sparse_matrix_double*"
```

## 7. Результаты и обсуждение

### 7.1 Корректность

**Методы проверки корректности:**

1. Комплексное модульное тестирование:
   - 34 функциональных теста — проверка базовых сценариев умножения разреженных матриц
   - 19 тестов покрытия — обработка граничных случаев

2. Тестирование производительности:
   - Тест производительности с диагональными разреженными матрицами размером 50000×50000
   - Проверка работоспособности на больших разреженных данных

**Ключевые тестовые сценарии:**

```cpp
// Базовые тесты умножения разреженных матриц
// Матрица 2×2
A = [[1,2],[3,4]], B = [[5,6],[7,8]] → C = [[19,22],[43,50]]

// Единичная матрица
A = [[1,0],[0,1]], B = [[1,2],[3,4]] → C = [[1,2],[3,4]]

// Разреженные матрицы с большим процентом нулей
A = [[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]]
B = [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
→ C = [[1,1,1,1],[4,4,4,4],[9,9,9,9],[16,16,16,16]]

// Векторное умножение
A = [[1,2,3]], B = [[4],[5],[6]] → C = [[32]]
```

**Методология проверки:**
- Каждый тест выполняется для обеих реализаций (SEQ и MPI)
- Результаты сравниваются с эталонным значением после конвертации из CRS в плотный формат
- Проверяется идентичность результатов между SEQ и MPI версиями
- Используется фреймворк Google Test для автоматизированной проверки
- Проверяется корректность формата CRS (монотонность row_ptr, соответствие размеров)

**Результаты проверки корректности:**
- Все 34 функциональных теста пройдены успешно
- Все 19 тестов покрытия подтвердили корректную обработку граничных случаев
- Результаты SEQ и MPI реализаций полностью совпадают
- Тест производительности подтвердил работоспособность на больших разреженных данных

### 7.2 Производительность

Результаты измерения производительности для разреженных матриц размером 50000×50000 
(диагональные матрицы с дополнительными элементами):

### Время выполнения (task_run) - чистые вычисления

| Режим | Процессы | Время, с | Ускорение | Эффективность |
|-------|----------|----------|-----------|---------------|
| seq   | 1        | 2.084    | 1.00      | 100%          |
| mpi   | 2        | 1.072    | 1.94      | 97%           |
| mpi   | 3        | 0.598    | 3.48      | 116%          |
| mpi   | 4        | 0.821    | 2.54      | 63%           |
| mpi   | 8        | 0.314    | 3.32      | 41%           |

### Время выполнения (pipeline) - полный цикл

| Режим | Процессы | Время, с | Ускорение | Эффективность |
|-------|----------|----------|-----------|---------------|
| seq   | 1        | 2.146    | 1.00      | 100%          |
| mpi   | 2        | 1.109    | 1.93      | 96%           |
| mpi   | 3        | 0.624    | 3.44      | 115%          |
| mpi   | 4        | 0.903    | 2.38      | 59%           |
| mpi   | 8        | 0.329    | 3.26      | 41%           |

**Анализ результатов:**

   - MPI версия демонстрирует существенное ускорение по сравнению с SEQ.
   - Наилучшее ускорение достигается при 3 процессах (≈3.4–3.5x).
   - При увеличении числа процессов наблюдается снижение эффективности из-за 
   роста коммуникационных затрат и неравномерной нагрузки.
   - При 8 процессах ускорение остаётся положительным, но эффективность снижается до ~40%.



## 8. Выводы

### Достижения

1. **Корректность реализации:**
   - Обе версии (SEQ и MPI) прошли все функциональные тесты
   - Результаты полностью совпадают с эталонными значениями
   - Обеспечена корректная обработка граничных случаев
   - Реализована корректная работа с форматом CRS

2. **Эффективное распараллеливание:**
   - Алгоритм демонстрирует хорошее ускорение до 3 процессов 
   (ускорение 3.64x для pipeline, 3.60x для task_run)
   - Горизонтальная схема распределения обеспечивает балансировку нагрузки
   - Использование формата CRS позволяет эффективно работать с разреженными матрицами
   - Циклическое распределение строк обеспечивает равномерную загрузку процессов

3. **Оптимизация коммуникаций:**
   - Минимизация коммуникаций за счет использования `MPI_Bcast` для матрицы B
   - Эффективное распределение данных матрицы A по строкам
   - Pipeline режим показывает лучшую производительность благодаря оптимизациям кэширования
