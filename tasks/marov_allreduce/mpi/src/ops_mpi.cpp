#include "marov_allreduce/mpi/include/ops_mpi.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

MPI_Comm::MPI_Comm(int r, int s) : rank(r), size(s) {
  parent.resize(size, -1);
  children.resize(size);
  buildTree();
}

void MPI_Comm::buildTree() {
  // Построение бинарного дерева
  for (int i = 0; i < size; i++) {
    int left = (2 * i) + 1;
    int right = (2 * i) + 2;

    if (left < size) {
      children[i].push_back(left);
      parent[left] = i;
    }
    if (right < size) {
      children[i].push_back(right);
      parent[right] = i;
    }
  }
}

void Send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
          MPI_Comm* comm) {
  (void)buf;
  (void)datatype;

  std::cout << "  [LOG] Process " << comm->rank << " -> " << dest
            << " (tag=" << tag << "): " << count << " elements"
            << "\n";

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

void Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag,
          MPI_Comm* comm, void* status) {
  (void)buf;
  (void)count;
  (void)datatype;
  (void)source;
  (void)tag;
  (void)comm;
  (void)status;

  std::cout << "  [LOG] Process " << comm->rank << " <- " << source
            << " (tag=" << tag << "): " << count << " elements"
            << "\n";

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

size_t getTypeSize(MPI_Datatype datatype) {
  switch (datatype) {
    case MPI_INT:
      return sizeof(int);
    case MPI_FLOAT:
      return sizeof(float);
    case MPI_DOUBLE:
      return sizeof(double);
    default:
      return 0;
  }
}

template <typename T>
void applyOperation(T* result, const T* data, int count, MPI_Op op) {
  switch (op) {
    case MPI_SUM:
      for (int i = 0; i < count; i++) {
        result[i] += data[i];
      }
      break;
    case MPI_MAX:
      for (int i = 0; i < count; i++) {
        if (data[i] > result[i]) {
          result[i] = data[i];
        }
      }
      break;
    case MPI_MIN:
      for (int i = 0; i < count; i++) {
        if (data[i] < result[i]) {
          result[i] = data[i];
        }
      }
      break;
    default:
      break;
  }
}

int my_allreduce(const void* sendbuf, void* recvbuf, int count,
                 MPI_Datatype datatype, MPI_Op op, MPI_Comm* comm) {
  int rank = comm->rank;
  size_t type_size = getTypeSize(datatype);

  // Копируем входные данные
  std::memcpy(recvbuf, sendbuf, count * type_size);

  // Фаза 1: Редукция (сбор данных к корню)
  for (int child : comm->children[rank]) {
    std::vector<char> child_buffer(count * type_size);

    Recv(child_buffer.data(), count, datatype, child, 0, comm, nullptr);

    switch (datatype) {
      case MPI_INT:
        applyOperation(static_cast<int*>(recvbuf),
                       reinterpret_cast<int*>(child_buffer.data()), count, op);
        break;
      case MPI_FLOAT:
        applyOperation(static_cast<float*>(recvbuf),
                       reinterpret_cast<float*>(child_buffer.data()), count,
                       op);
        break;
      case MPI_DOUBLE:
        applyOperation(static_cast<double*>(recvbuf),
                       reinterpret_cast<double*>(child_buffer.data()), count,
                       op);
        break;
      default:
        break;
    }
  }

  // Отправляем результат родителю (если не корень)
  if (rank != 0) {
    Send(recvbuf, count, datatype, comm->parent[rank], 0, comm);
  }

  // Фаза 2: Рассылка (broadcast результата всем)
  if (rank != 0) {
    Recv(recvbuf, count, datatype, comm->parent[rank], 1, comm, nullptr);
  }

  // Отправляем результат детям
  for (int child : comm->children[rank]) {
    Send(recvbuf, count, datatype, child, 1, comm);
  }

  return 0;
}
