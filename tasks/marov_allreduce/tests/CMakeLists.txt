add_executable(marov_allreduce_tests functional/main.cpp)

target_link_libraries(marov_allreduce_tests
    gtest
    gtest_main
)

target_include_directories(marov_allreduce_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../mpi/include
)
