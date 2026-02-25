#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "ops_seq.h"

TEST(radix_sort_double_test, basic_sort) {
  std::vector<double> arr = {3.14, 1.41, 2.71, 0.58, 1.61};
  std::vector<double> expected = arr;
  std::sort(expected.begin(), expected.end());

  radix_sort_double(arr);

  EXPECT_EQ(arr, expected);
}

TEST(radix_sort_double_test, negative_numbers) {
  std::vector<double> arr = {-2.5, 3.7, -1.1, 0.0, 4.2, -3.3};
  std::vector<double> expected = arr;
  std::sort(expected.begin(), expected.end());

  radix_sort_double(arr);

  EXPECT_EQ(arr, expected);
}
