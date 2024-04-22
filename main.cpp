#include <windows.h>
#include <string>
#include <string_view>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

std::string TestString(size_t len)
{
    std::string result;
    result.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        result.push_back(static_cast<char>(i % 256));
    }
    return result;
}

uint32_t ComputeCrc32w(std::string_view str)
{
    return RtlCrc32(str.data(), str.size(), 0);
}

uint32_t ComputeCrc32c(std::string_view str);

int main()
{
    auto test = TestString(256);
    auto res_w = ComputeCrc32w(test);
    auto res_c = ComputeCrc32c(test);
    if (res_w != res_c)
        printf("ERROR: res_w(0x%08x) != res_c(0x%08x)\n", res_w, res_c);
    else
        printf("OK\n");
}

void donothing(uint64_t x)
{
    // printf("x:%llx\n", x);
}
