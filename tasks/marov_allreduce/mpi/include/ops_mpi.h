#pragma once

#include <vector>

// Типы данных MPI
typedef int MPI_Datatype;
typedef int MPI_Op;

#define MPI_INT 1
#define MPI_FLOAT 2
#define MPI_DOUBLE 3
#define MPI_SUM 1
#define MPI_MAX 2
#define MPI_MIN 3

struct MPI_Comm {
  int rank;
  int size;
  std::vector<int> parent;
  std::vector<std::vector<int>> children;

  MPI_Comm(int r, int s);
  void buildTree();
};

void Send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
          MPI_Comm* comm);

void Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag,
          MPI_Comm* comm, void* status);

size_t getTypeSize(MPI_Datatype datatype);

template <typename T>
void applyOperation(T* result, const T* data, int count, MPI_Op op);

int my_allreduce(const void* sendbuf, void* recvbuf, int count,
                 MPI_Datatype datatype, MPI_Op op, MPI_Comm* comm);
