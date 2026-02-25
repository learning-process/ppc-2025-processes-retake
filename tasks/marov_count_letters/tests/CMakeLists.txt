add_executable(marov_count_letters_tests functional/ops_seq_tests.cpp)

target_link_libraries(marov_count_letters_tests
    gtest
    gtest_main
    pthread
)

target_include_directories(marov_count_letters_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
