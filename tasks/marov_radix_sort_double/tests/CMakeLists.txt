add_executable(marov_radix_sort_double_tests functional/ops_seq_tests.cpp)

target_link_libraries(marov_radix_sort_double_tests
    gtest
    gtest_main
    pthread
)

target_include_directories(marov_radix_sort_double_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
