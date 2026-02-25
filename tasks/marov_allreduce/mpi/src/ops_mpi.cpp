#include "ops_mpi.h"

#include <cstring>
#include <vector>

int my_allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int type_size;
    MPI_Type_size(datatype, &type_size);

    std::vector<char> buffer(count * type_size);
    std::memcpy(buffer.data(), sendbuf, count * type_size);

    // Сбор данных на root
    if (rank == 0)
    {
        std::memcpy(recvbuf, buffer.data(), count * type_size);
        for (int i = 1; i < size; i++)
        {
            MPI_Recv(buffer.data(), count, datatype, i, 0, comm, MPI_STATUS_IGNORE);

            // Применяем операцию op
            if (op == MPI_SUM)
            {
                for (int j = 0; j < count; j++)
                {
                    if (datatype == MPI_INT)
                    {
                        ((int *) recvbuf)[j] += ((int *) buffer.data())[j];
                    }
                    else if (datatype == MPI_FLOAT)
                    {
                        ((float *) recvbuf)[j] += ((float *) buffer.data())[j];
                    }
                    else if (datatype == MPI_DOUBLE)
                    {
                        ((double *) recvbuf)[j] += ((double *) buffer.data())[j];
                    }
                }
            }
            else if (op == MPI_MAX)
            {
                for (int j = 0; j < count; j++)
                {
                    if (datatype == MPI_INT)
                    {
                        if (((int *) buffer.data())[j] > ((int *) recvbuf)[j])
                            ((int *) recvbuf)[j] = ((int *) buffer.data())[j];
                    }
                    else if (datatype == MPI_FLOAT)
                    {
                        if (((float *) buffer.data())[j] > ((float *) recvbuf)[j])
                            ((float *) recvbuf)[j] = ((float *) buffer.data())[j];
                    }
                    else if (datatype == MPI_DOUBLE)
                    {
                        if (((double *) buffer.data())[j] > ((double *) recvbuf)[j])
                            ((double *) recvbuf)[j] = ((double *) buffer.data())[j];
                    }
                }
            }
            // Добавь другие операции по необходимости
        }
        // Рассылка результата
        for (int i = 1; i < size; i++)
        {
            MPI_Send(recvbuf, count, datatype, i, 0, comm);
        }
    }
    else
    {
        MPI_Send(buffer.data(), count, datatype, 0, 0, comm);
        MPI_Recv(recvbuf, count, datatype, 0, 0, comm, MPI_STATUS_IGNORE);
    }

    return MPI_SUCCESS;
}
