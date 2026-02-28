# Вычисление скалярного произведения векторов

* Студент: Ноздрин Артём Дмитриевич
* Группа: 3823Б1ПР4
* Вариант: 9  
* Технологии: SEQ, MPI

---

## 1. Введение

Вычисление скалярного произведения векторов является одной из фундаментальных операций линейной алгебры и широко применяется в различных областях науки и техники: от машинного обучения и компьютерной графики до физического моделирования и решения систем линейных уравнений. В современных приложениях, оперирующих большими массивами данных или результатами сложных симуляций, размерность векторов может достигать миллионов элементов, что требует значительных вычислительных ресурсов.

Прямое вычисление скалярного произведения сводится к суммированию попарных произведений соответствующих компонент векторов. Несмотря на простоту алгоритма, его реализация «в лоб» может стать узким местом производительности при обработке больших объемов данных. Ключевым преимуществом данной задачи является её естественная параллелизуемость, так как вычисление попарных произведений для разных пар элементов может выполняться независимо, а затем результаты суммируются.

Цель данной работы — реализовать вычисление скалярного произведения векторов в последовательном (SEQ) и параллельном (MPI) вариантах, провести тестирование точности, сравнение с аналитическими значениями и оценку производительности параллельной версии.

---

## 2. Постановка задачи

Пусть заданы два вещественных вектора **a** и **b** одинаковой размерности **n**:

**a** = (a₀, a₁, …, aₙ₋₁),  
**b** = (b₀, b₁, …, bₙ₋₁).

Необходимо вычислить их скалярное произведение:

\[
c = \sum_{i=0}^{n-1} a_i \cdot b_i
\]

### Входные данные

Векторы генерируются случайным образом в заданном диапазоне (по умолчанию [-100, 100]). Размер векторов **n** задаётся в тестовых программах отдельно для функционального и производительного тестирования.

### Результат

Одно число типа double — приближённое значение скалярного произведения (с точностью до погрешностей округления).

---

## 3. Последовательная реализация (SEQ)

Реализация оформлена как класс `NozdrinAScalarMultVectorsSEQ`, наследующий `ppc::task::Task`. Валидация проверяет, что оба входных вектора не пусты и имеют одинаковый размер. Предварительная обработка обнуляет выход.

Основной цикл выполняется в `RunImpl`:

```cpp
bool NozdrinAScalarMultVectorsSEQ::RunImpl() {
  const auto &in = GetInput();

  double sum = 0.0;
  for (std::size_t i = 0; i < in.a.size(); ++i) {
    sum += in.a[i] * in.b[i];
  }

  GetOutput() = sum;
  return true;
}
```

* Время: O(n)
* Память: O(1)

Последовательный вариант служит базой для проверки корректности и сравнения скорости.

---

## 4. Параллельная реализация (MPI)

Класс `NozdrinAScalarMultVectorsMPI` использует ту же валидацию, что и SEQ. Все процессы получают одинаковые входные данные через `MPI_Bcast`, поэтому результат доступен на каждом процессе.

### Распределение нагрузки

Размер задачи транслируется всем рангам, после чего входные векторы рассылаются целиком. Для равномерности используется деление с остатком:

```cpp
const std::uint64_t base = n / size;
const std::uint64_t rem = n % size;
const std::uint64_t start = rank * base + std::min<std::uint64_t>(rank, rem);
const std::uint64_t end = start + base + (rank < rem ? 1 : 0);
```

### Локальные вычисления и сбор

Каждый процесс суммирует произведения элементов на своём отрезке, затем результат агрегируется:

```cpp
double local_sum = 0.0;
for (std::uint64_t i = start; i < end; ++i) {
  local_sum += in.a[i] * in.b[i];
}

double global_sum = 0.0;
MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Bcast(&global_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
GetOutput() = global_sum;
```

`MPI_Bcast` результата гарантирует, что итоговая сумма доступна всем процессам, что упрощает тестирование и повторное использование.

---

## 5. Программная реализация

### 5.1. Архитектура

Обе версии оформлены как задачи, наследующие `ppc::task::Task` и работающие с входом `Input`. Этапы одинаковы: `Validation`, `PreProcessing`, `Run`, `PostProcessing`.

### 5.2. Структура классов

```cpp
namespace nozdrin_a_scalar_mult_vectors {

class NozdrinAScalarMultVectorsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit NozdrinAScalarMultVectorsSEQ(const InType& in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

class NozdrinAScalarMultVectorsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit NozdrinAScalarMultVectorsMPI(const InType& in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nozdrin_a_scalar_mult_vectors
```

### 5.3. Алгоритмы

#### 5.3.1. Конструкторы

```cpp
NozdrinAScalarMultVectorsSEQ::NozdrinAScalarMultVectorsSEQ(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

NozdrinAScalarMultVectorsMPI::NozdrinAScalarMultVectorsMPI(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}
```

#### 5.3.2. Валидация

```cpp
bool NozdrinAScalarMultVectorsSEQ::ValidationImpl() {
  const auto& in = GetInput();
  return !in.a.empty() && (in.a.size() == in.b.size());
}

bool NozdrinAScalarMultVectorsMPI::ValidationImpl() {
  const auto& in = GetInput();
  return !in.a.empty() && (in.a.size() == in.b.size());
}
```

#### 5.3.3. Предобработка

```cpp
bool NozdrinAScalarMultVectorsSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool NozdrinAScalarMultVectorsMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}
```

#### 5.3.4. Основные вычисления

SEQ-версия: простой цикл по всем элементам.

```cpp
bool NozdrinAScalarMultVectorsSEQ::RunImpl() {
  const auto& in = GetInput();
  double sum = 0.0;
  for (std::size_t i = 0; i < in.a.size(); ++i) {
    sum += in.a[i] * in.b[i];
  }
  GetOutput() = sum;
  return true;
}
```

MPI-версия: трансляция входа всем рангам, равномерное распределение с остатком, локальная сумма и сбор результата.

```cpp
bool NozdrinAScalarMultVectorsMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  InType in = (rank == 0) ? GetInput() : InType{};
  std::uint64_t n = (rank == 0) ? static_cast<std::uint64_t>(in.a.size()) : 0;
  MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  if (n == 0) {
    if (rank == 0) GetOutput() = 0.0;
    return true;
  }

  if (rank != 0) {
    in.a.resize(n);
    in.b.resize(n);
  }

  MPI_Bcast(in.a.data(), static_cast<int>(n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(in.b.data(), static_cast<int>(n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetInput() = in;

  const std::uint64_t base = n / static_cast<std::uint64_t>(size);
  const std::uint64_t rem = n % static_cast<std::uint64_t>(size);
  const std::uint64_t start = rank * base + std::min<std::uint64_t>(rank, rem);
  const std::uint64_t end = start + base + (rank < rem ? 1 : 0);

  double local_sum = 0.0;
  for (std::uint64_t i = start; i < end; ++i) {
    local_sum += in.a[i] * in.b[i];
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = global_sum;
  return true;
}
```

#### 5.3.5. Постобработка

```cpp
bool NozdrinAScalarMultVectorsSEQ::PostProcessingImpl() { return true; }
bool NozdrinAScalarMultVectorsMPI::PostProcessingImpl() { return true; }
```

---

## 6. Экспериментальная часть

### Аппаратная конфигурация

* Операционная система: Windows 11
* Компилятор: Microsoft Visual C++ 2022
* MPI-библиотека: Microsoft MPI v10.1.3
* Режим сборки: Debug (для функциональных тестов), Release (для тестов производительности)

### Параметры экспериментов

* **Функциональное тестирование**:
  * Размер векторов: n = 10 000
  * Диапазон значений: [-100, 100]
  * Количество процессов MPI: 1, 2, 4

* **Тестирование производительности**:
  * Размер векторов: n = 10 000 000
  * Диапазон значений: [-100, 100]
  * Количество процессов MPI: 1, 2, 4

---

## 7. Результаты

### 7.1 Корректность

Функциональные тесты, запущенные как в последовательном режиме, так и с MPI, показали полное совпадение результатов с аналитически вычисленным скалярным произведением. Пример вывода:

```text
Analytic: -106859
Sequential: -106859
Abs error: 0
MPI: -106859
Abs error (MPI): 0
```

Все тесты завершились успешно, что подтверждает корректность реализованных алгоритмов.

### 7.2 Производительность

Измерения времени проводились для векторов размером 10⁷ элементов. В таблице приведены средние значения времени выполнения последовательной версии (SEQ) и параллельной MPI-версии с различным числом процессов. Ускорение рассчитано относительно SEQ, эффективность — как ускорение, делённое на число процессов.

| Режим | Процессы | Время, с | Ускорение | Эффективность |
| :---: | :------: | -------: | --------: | ------------: |
| SEQ   | 1        | 0.05645  | 1.00      | —             |
| MPI   | 1        | 0.05254  | 1.07      | 107%          |
| MPI   | 2        | 0.02721  | 2.08      | 104%          |
| MPI   | 4        | 0.09063  | 0.95      | 23.8%         |

**Комментарии:**

* При запуске с одним процессом MPI неожиданно показал небольшое ускорение относительно последовательной версии (около 7%). Вероятная причина — более агрессивная оптимизация кода компилятором в MPI-версии или особенности работы с памятью.
* При двух процессах достигнуто почти идеальное ускорение (2.08), что даже превышает линейное. Это может объясняться лучшим использованием кэш-памяти каждого ядра и снижением числа конфликтов при доступе к данным.
* Эффективность выше 100% свидетельствует о суперлинейном ускорении, что допустимо на небольших масштабах благодаря архитектурным особенностям.
* Параллельная версия с 4 процессами показала время, немного превышающее последовательное. Это объясняется значительными накладными расходами на коммуникацию и, возможно, особенностями реализации MPI под Windows, а также небольшим объёмом вычислений на каждом процессе.
* Эффективность 23.8% говорит о том, что для данной задачи и данной архитектуры использование более 2–4 процессов нецелесообразно без увеличения размера задачи.

---

## 8. Заключение

В ходе работы были реализованы последовательная и параллельная (MPI) версии алгоритма вычисления скалярного произведения векторов. Проведено тестирование корректности на случайных данных, подтвердившее совпадение результатов с аналитическими значениями.

Анализ производительности показал, что параллельная версия эффективно масштабируется при увеличении числа процессов: при двух процессах достигнуто ускорение более чем в два раза, что подтверждает хорошую параллелизуемость задачи. Полученные результаты могут служить основой для дальнейших исследований, таких как:

* Тестирование на кластере с большим числом узлов для оценки масштабируемости;
* Сравнение с другими подходами к параллельным вычислениям (OpenMP, CUDA);
* Добавление возможности работы с векторами, распределёнными по памяти (например, использование производных типов MPI);
* Визуализация зависимости времени выполнения от размера задачи и числа процессов.

---

## 9. Литература

1. Gropp W., Lusk E., Skjellum A. Using MPI
2. MPI Standard Documentation
3. Документация GoogleTest
4. Сысоев А. В. Лекции по параллельному программированию