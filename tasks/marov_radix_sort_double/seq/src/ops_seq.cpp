#include "ops_seq.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

void radix_sort_double(std::vector<double> &arr)
{
    if (arr.empty())
    {
        return;
    }

    std::vector<uint64_t> bits(arr.size());
    for (size_t i = 0; i < arr.size(); ++i)
    {
        uint64_t val = 0;
        std::memcpy(&val, &arr[i], sizeof(double));
        if ((val >> 63) != 0)
        {
            val = ~val;
        }
        else
        {
            val |= (1ULL << 63);
        }
        bits[i] = val;
    }

    const int bits_per_pass = 8;
    const int radix = 1 << bits_per_pass;
    const int passes = sizeof(uint64_t) * 8 / bits_per_pass;

    std::vector<uint64_t> temp(arr.size());

    for (int pass = 0; pass < passes; ++pass)
    {
        std::vector<int> count(radix, 0);

        for (size_t i = 0; i < bits.size(); ++i)
        {
            int digit = static_cast<int>((bits[i] >> (pass * bits_per_pass)) & (radix - 1));
            ++count[digit];
        }

        for (int i = 1; i < radix; ++i)
        {
            count[i] += count[i - 1];
        }

        for (int i = static_cast<int>(bits.size()) - 1; i >= 0; --i)
        {
            int digit = static_cast<int>((bits[i] >> (pass * bits_per_pass)) & (radix - 1));
            temp[--count[digit]] = bits[i];
        }

        bits.swap(temp);
    }

    for (size_t i = 0; i < arr.size(); ++i)
    {
        uint64_t val = bits[i];
        if ((val >> 63) != 0)
        {
            val &= ~(1ULL << 63);
        }
        else
        {
            val = ~val;
        }
        std::memcpy(&arr[i], &val, sizeof(double));
    }
}
