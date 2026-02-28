cmake_minimum_required(VERSION 3.10)
project(scalar_multiplication)
set(CMAKE_CXX_STANDARD 17)

include_directories(common.include)
include_directories(seq/include)
include_directories(mpi/include)
include_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
link_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")

add_library(common STATIC common.include/common.cpp)
add_library(seq STATIC seq/src/ops_seq.cpp)
add_library(mpi_ops STATIC mpi/src/ops_mpi.cpp)

add_executable(functional_test tests/functional/main.cpp)
target_link_libraries(functional_test common seq)

add_executable(performance_test tests/performance/main.cpp)
target_link_libraries(performance_test common seq)

find_package(MPI)
if(MPI_FOUND)
    target_compile_definitions(functional_test PRIVATE USE_MPI)
    target_link_libraries(functional_test mpi_ops MPI::MPI_CXX)
    target_compile_definitions(performance_test PRIVATE USE_MPI)
    target_link_libraries(performance_test mpi_ops MPI::MPI_CXX)
endif()
