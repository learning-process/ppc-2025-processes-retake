#ifndef OPS_MPI_H
#define OPS_MPI_H

#include <mpi.h>

int my_allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

#endif
