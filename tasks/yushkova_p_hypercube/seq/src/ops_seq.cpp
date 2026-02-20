#include "yushkova_p_hypercube/seq/include/ops_seq.hpp"

#include <mpi.h>

#include "yushkova_p_hypercube/common/include/common.hpp"

namespace yushkova_p_hypercube {

namespace {

bool IsPowerOfTwo(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

}  // namespace

YushkovaPHypercubeSEQ::YushkovaPHypercubeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YushkovaPHypercubeSEQ::ValidationImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int source = std::get<0>(GetInput());
  const int destination = std::get<1>(GetInput());
  const bool correct_source = source >= 0 && source < world_size;
  const bool correct_destination = destination >= 0 && destination < world_size;
  return IsPowerOfTwo(world_size) && correct_source && correct_destination;
}

bool YushkovaPHypercubeSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YushkovaPHypercubeSEQ::RunImpl() {
  const int source = std::get<0>(GetInput());
  const int destination = std::get<1>(GetInput());
  const int payload = std::get<2>(GetInput());

  int current_owner = source;
  const int value = payload;
  const int route_mask = source ^ destination;

  for (int bit = 0; (1 << bit) <= route_mask; ++bit) {
    const bool bit_differs = (route_mask & (1 << bit)) != 0;
    if (bit_differs) {
      current_owner ^= (1 << bit);
    }
  }

  GetOutput() = (current_owner == destination) ? value : 0;
  return true;
}

bool YushkovaPHypercubeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace yushkova_p_hypercube
