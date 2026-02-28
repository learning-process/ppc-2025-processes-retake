add_executable(marov_count_letters_tests functional/main.cpp)

target_link_libraries(marov_count_letters_tests
    gtest
    gtest_main
    MPI::MPI_CXX
)

target_include_directories(marov_count_letters_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../seq/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../mpi/include
)
