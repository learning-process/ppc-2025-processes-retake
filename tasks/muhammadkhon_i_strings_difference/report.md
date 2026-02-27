## Введение

В рамках лабораторной работы была реализована последовательная и
параллельная версия алгоритма вычисления количества различающихся
символов в двух строках.

Цель работы — изучить основы параллельного программирования с
использованием MPI, реализовать декомпозицию задачи по данным и
сравнить производительность последовательной и MPI-версии.

## Постановка задачи

Даны две строки s1 и s2. Необходимо вычислить количество позиций, в
которых символы различаются.

Если строки имеют разную длину, то дополнительно учитывается разница
длин.

Ограничения:

- строки могут быть большой длины;
- решение должно корректно работать при любом количестве
MPI-процессов;
- необходимо обеспечить одинаковый результат в SEQ и MPI версиях.

# Описание алгоритма

Последовательный алгоритм реализован следующим образом:

1. Определяется минимальная и максимальная длина строк.
2. Выполняется проход по инидексам от 0 до min_len - 1.
3. Подсчитываются позиции где символы различаются.
4. К результаты под конец добавляется разница длины строк.

собственно сам код:

``` C++
bool MuhammadkhonIStringsDifferenceSEQ::RunImpl() {
  const auto &[s1, s2] = GetInput();

  size_t min_len = std::min(s1.size(), s2.size());
  size_t max_len = std::max(s1.size(), s2.size());

  int diff_count = 0;

  for (size_t i = 0; i < min_len; ++i) {
    if (s1[i] != s2[i]) {
      diff_count++;
    }
  }
  diff_count += static_cast<int>(max_len - min_len);

  GetOutput() = diff_count;
  return true;
}
```

# Схема распараллеливания (MPI)

Использована блочная декомпозиция по данным.

Обрабатывается только общая часть строк (min_len).

Она делится между процессами следующим образом:

- каждый процесс получает min_len/proc_size
- остаток rem = min_len % proc_size
- первые rem процессов получают на 1 символ больше

Таким образом обеспечивается равномерное распределение нагрузки

``` C++
 if (proc_rank_ == 0) {
    size_t offset = 0;
    for (int i = 0; i < proc_size_; ++i) {
      send_counts[i] = static_cast<int>(base + (std::cmp_less(i, rem) ? 1 : 0));
      displs[i] = static_cast<int>(offset);
      offset += send_counts[i];
    }
  }

  MPI_Bcast(send_counts.data(), proc_size_, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), proc_size_, MPI_INT, 0, MPI_COMM_WORLD);

  const size_t local_size = base + (std::cmp_less(proc_rank_, rem) ? 1 : 0);

  std::vector<char> local_s1(local_size);
  std::vector<char> local_s2(local_size);

  MPI_Scatterv(proc_rank_ == 0 ? const_cast<char *>(s1.data()) : nullptr, send_counts.data(), displs.data(), MPI_CHAR,
               local_s1.data(), static_cast<int>(local_size), MPI_CHAR, 0, MPI_COMM_WORLD);

  MPI_Scatterv(proc_rank_ == 0 ? const_cast<char *>(s2.data()) : nullptr, send_counts.data(), displs.data(), MPI_CHAR,
               local_s2.data(), static_cast<int>(local_size), MPI_CHAR, 0, MPI_COMM_WORLD);

  int local_diff = 0;
  for (size_t i = 0; i < local_size; ++i) {
    if (local_s1[i] != local_s2[i]) {
      local_diff++;
    }
  }

  int global_diff = 0;
  MPI_Allreduce(&local_diff, &global_diff, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = global_diff + static_cast<int>(max_len - min_len);
```

# Коммуникационная схема

1. Процесс 0:

- получает входные строки;
- вычисляет длины;
- формирует массивы send_counts и displs.

2. Выполняются коллективные операции:

- MPI_Bcast — передача длин строк;
- MPI_Scatterv — распределение фрагментов строк;
- локальный подсчёт различий;
- MPI_Allreduce с операцией MPI_SUM — суммирование локальных
результатов.

Дополнительно учитывается разница длин строк.

# Экперементальные результаты

Окружение

- ОС: Linux(Fedora)
- Компилятор: g++
- Сборка: Release
- MPI: OpenMPI

Результаты производительности

| Версия | Режим    | Время (сек) |
| ------ | -------- | ----------- |
| MPI    | pipeline | 0.4679      |
| MPI    | task_run | 0.4531      |
| SEQ    | pipeline | 0.0810      |
| SEQ    | task_run | 0.0836      |

# Выводы

Что показал эксперимент:

- для задач с линейной сложностью и минимальной вычислительной
нагрузкой MPI не даёт выигрыша;
- накладные расходы на коммуникации доминируют над вычислениями;
- параллелизация эффективна только при достаточно тяжёлых
вычислениях.

Вывод:
MPI оправдан для вычислительно интенсивных задач, но не для простых
линейных операций над строками.
