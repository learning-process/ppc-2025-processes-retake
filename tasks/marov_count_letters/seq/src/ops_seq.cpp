#include "ops_seq.h"
#include <cctype>

int count_letters(const std::string& str) {
    int count = 0;
    for (char c : str) {
        if (std::isalpha(c)) {
            count++;
        }
    }
    return count;
}
