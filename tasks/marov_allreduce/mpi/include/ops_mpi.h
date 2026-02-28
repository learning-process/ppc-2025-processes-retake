#pragma once

#include <cstddef>
#include <vector>

// MPI-style types for compatibility with MPI API
using MpiDatatype = int;
using MpiOp = int;

#define MPI_INT 1
#define MPI_FLOAT 2
#define MPI_DOUBLE 3
#define MPI_SUM 1
#define MPI_MAX 2
#define MPI_MIN 3

struct MpiComm {
  int rank;
  int size;
  std::vector<int> parent;
  std::vector<std::vector<int>> children;

  MpiComm(int r, int s);
  void BuildTree();
};

void Send(const void* buf, int count, MpiDatatype datatype, int dest, int tag,
          MpiComm* comm);

void Recv(void* buf, int count, MpiDatatype datatype, int source, int tag,
          MpiComm* comm, void* status);

size_t GetTypeSize(MpiDatatype datatype);

template <typename T>
void ApplyOperation(T* result, const T* data, int count, MpiOp op);

int MyAllreduce(const void* sendbuf, void* recvbuf, int count,
                MpiDatatype datatype, MpiOp op, MpiComm* comm);
