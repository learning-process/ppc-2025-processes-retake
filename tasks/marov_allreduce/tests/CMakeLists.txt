add_executable(marov_allreduce_tests functional/ops_mpi_tests.cpp)

target_link_libraries(marov_allreduce_tests
    gtest
    gtest_main
    pthread
    MPI::MPI_CXX
)

target_include_directories(marov_allreduce_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
