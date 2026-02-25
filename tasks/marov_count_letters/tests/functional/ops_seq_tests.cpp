#include <gtest/gtest.h>

#include "../../seq/src/ops_seq.h"

TEST(count_letters_seq, basic)
{
    std::string str = "Hello, world!";
    EXPECT_EQ(count_letters(str), 10);
}

TEST(count_letters_seq, no_letters)
{
    std::string str = "12345!@#$%";
    EXPECT_EQ(count_letters(str), 0);
}

TEST(count_letters_seq, empty)
{
    std::string str = "";
    EXPECT_EQ(count_letters(str), 0);
}
