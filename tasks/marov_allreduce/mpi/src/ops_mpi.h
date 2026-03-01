#ifndef OPS_MPI_H
#define OPS_MPI_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <cstring>

// Типы данных MPI
enum MPI_Datatype {
    MPI_INT,
    MPI_FLOAT,
    MPI_DOUBLE
};

// Операции редукции
enum MPI_Op {
    MPI_SUM,
    MPI_MAX,
    MPI_MIN
};

struct MPI_Comm {
    int rank;
    int size;
    std::vector<int> parent;
    std::vector<std::vector<int>> children;

    MPI_Comm(int r, int s);
    void buildTree();
};

// Функции Send/Recv
void Send(const void* buf, int count, MPI_Datatype datatype, int dest,
          int tag, MPI_Comm* comm);

void Recv(void* buf, int count, MPI_Datatype datatype, int source,
          int tag, MPI_Comm* comm, void* status = nullptr);

// Allreduce через дерево
int my_allreduce(const void* sendbuf, void* recvbuf, int count,
                 MPI_Datatype datatype, MPI_Op op, MPI_Comm* comm);

#endif
