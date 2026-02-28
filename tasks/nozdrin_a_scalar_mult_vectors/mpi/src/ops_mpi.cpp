#include "nozdrin_a_scalar_mult_vectors/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>

#include "nozdrin_a_scalar_mult_vectors/common/include/common.hpp"

namespace nozdrin_a_scalar_mult_vectors {

NozdrinAScalarMultVectorsMPI::NozdrinAScalarMultVectorsMPI(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
    GetOutput() = 0.0;
}

bool NozdrinAScalarMultVectorsMPI::ValidationImpl() {
    const auto &in = GetInput();
    return !in.a.empty() && (in.a.size() == in.b.size());
}

bool NozdrinAScalarMultVectorsMPI::PreProcessingImpl() {
    GetOutput() = 0.0;
    return true;
}

bool NozdrinAScalarMultVectorsMPI::RunImpl() {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    InType in = (rank == 0) ? GetInput() : InType{};

        std::uint64_t n = (rank == 0) ? static_cast<std::uint64_t>(in.a.size()) : 0;
        MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    if (n == 0) {
        if (rank == 0) {
            GetOutput() = 0.0;
        }
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

    const std::uint64_t start = (static_cast<std::uint64_t>(rank) * base) + std::min<std::uint64_t>(rank, rem);
    const std::uint64_t end = start + base + (static_cast<std::uint64_t>(rank) < rem ? 1 : 0);

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

bool NozdrinAScalarMultVectorsMPI::PostProcessingImpl() {
    return true;
}

}  // namespace nozdrin_a_scalar_mult_vectors
