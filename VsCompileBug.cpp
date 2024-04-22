#include <stdint.h>
#include <string_view>
#if defined(__SSE4_2__) && defined(__PCLMUL__)
#include <x86intrin.h>
#elif defined(_MSC_VER) && defined(__AVX__)
#include <intrin.h>
#endif
typedef __m128i V128;

void donothing(uint64_t x);

static __forceinline void PrefetchToLocalCache(const void* addr)
{
#ifdef _MSC_VER
    _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0);
#else
    __builtin_prefetch(addr, 0, 3);
#endif
}

static __forceinline uint32_t CRC32_u8(uint32_t crc, uint8_t v)
{
    return _mm_crc32_u8(crc, v);
}

static __forceinline uint32_t CRC32_u16(uint32_t crc, uint16_t v)
{
    return _mm_crc32_u16(crc, v);
}

static __forceinline uint32_t CRC32_u32(uint32_t crc, uint32_t v)
{
    return _mm_crc32_u32(crc, v);
}

static __forceinline uint32_t CRC32_u64(uint32_t crc, uint64_t v)
{
#if defined(__x86_64__) || defined(_M_X64)
    return static_cast<uint32_t>(_mm_crc32_u64(crc, v));
#else
    uint32_t v_lo = static_cast<uint32_t>(v);
    uint32_t v_hi = static_cast<uint32_t>(v >> 32);
    return _mm_crc32_u32(_mm_crc32_u32(crc, v_lo), v_hi);
#endif
}

static __forceinline V128 V128_From64WithZeroFill(const uint64_t r)
{
    return _mm_set_epi64x(static_cast<int64_t>(0), static_cast<int64_t>(r));
}

static __forceinline V128 V128_PMulLow(const V128 l, const V128 r)
{
    return _mm_clmulepi64_si128(l, r, 0x00);
}

static __forceinline V128 V128_PMul10(const V128 l, const V128 r)
{
    return _mm_clmulepi64_si128(l, r, 0x10);
}

static __forceinline V128 V128_Xor(const V128 l, const V128 r)
{
    return _mm_xor_si128(l, r);
}

static __forceinline int64_t V128_Low64(const V128 l)
{
#if defined(__x86_64__) || defined(_M_X64)
    return _mm_cvtsi128_si64(l);
#else
    uint32_t r_lo = static_cast<uint32_t>(_mm_extract_epi32(l, 0));
    uint32_t r_hi = static_cast<uint32_t>(_mm_extract_epi32(l, 1));
    return static_cast<int64_t>((static_cast<uint64_t>(r_hi) << 32) | r_lo);
#endif
}

#define ABSL_INTERNAL_STEP1(crc)                                         \
  do {                                                                   \
    crc = CRC32_u8(static_cast<uint32_t>(crc), *(uint8_t*)(void*)(p++)); \
  } while (0)
#define ABSL_INTERNAL_STEP2(crc)                                         \
  do {                                                                   \
    crc = CRC32_u16(static_cast<uint32_t>(crc), *(uint16_t*)(void*)(p)); \
    p += 2;                                                              \
  } while (0)
#define ABSL_INTERNAL_STEP4(crc)                                         \
  do {                                                                   \
    crc = CRC32_u32(static_cast<uint32_t>(crc), *(uint32_t*)(void*)(p)); \
    p += 4;                                                              \
  } while (0)
#define ABSL_INTERNAL_STEP8(crc, p)                                      \
  do {                                                                   \
    crc = CRC32_u64(static_cast<uint32_t>(crc), *(uint64_t*)(void*)(p)); \
    p += 8;                                                              \
  } while (0)
#define ABSL_INTERNAL_STEP8BY2(crc0, crc1, p0, p1) \
  do {                                             \
    ABSL_INTERNAL_STEP8(crc0, p0);                 \
    ABSL_INTERNAL_STEP8(crc1, p1);                 \
  } while (0)
#define ABSL_INTERNAL_STEP8BY3(crc0, crc1, crc2, p0, p1, p2) \
  do {                                                       \
    ABSL_INTERNAL_STEP8(crc0, p0);                           \
    ABSL_INTERNAL_STEP8(crc1, p1);                           \
    ABSL_INTERNAL_STEP8(crc2, p2);                           \
  } while (0)

static constexpr size_t kGroupsSmall = 3;
static constexpr size_t kSmallCutoff = 256;
static constexpr int kPrefetchHorizonMedium = 64;

alignas(16) static constexpr uint64_t kClmulConstants[] = {
    0x09e4addf8, 0x0ba4fc28e, 0x00d3b6092, 0x09e4addf8, 0x0ab7aff2a,
    0x102f9b8a2, 0x0b9e02b86, 0x00d3b6092, 0x1bf2e8b8a, 0x18266e456,
    0x0d270f1a2, 0x0ab7aff2a, 0x11eef4f8e, 0x083348832, 0x0dd7e3b0c,
    0x0b9e02b86, 0x0271d9844, 0x1b331e26a, 0x06b749fb2, 0x1bf2e8b8a,
    0x0e6fc4e6a, 0x0ce7f39f4, 0x0d7a4825c, 0x0d270f1a2, 0x026f6a60a,
    0x12ed0daac, 0x068bce87a, 0x11eef4f8e, 0x1329d9f7e, 0x0b3e32c28,
    0x0170076fa, 0x0dd7e3b0c, 0x1fae1cc66, 0x010746f3c, 0x086d8e4d2,
    0x0271d9844, 0x0b3af077a, 0x093a5f730, 0x1d88abd4a, 0x06b749fb2,
    0x0c9c8b782, 0x0cec3662e, 0x1ddffc5d4, 0x0e6fc4e6a, 0x168763fa6,
    0x0b0cd4768, 0x19b1afbc4, 0x0d7a4825c, 0x123888b7a, 0x00167d312,
    0x133d7a042, 0x026f6a60a, 0x000bcf5f6, 0x19d34af3a, 0x1af900c24,
    0x068bce87a, 0x06d390dec, 0x16cba8aca, 0x1f16a3418, 0x1329d9f7e,
    0x19fb2a8b0, 0x02178513a, 0x1a0f717c4, 0x0170076fa,
};

static __forceinline void Extend(uint32_t* crc, const void* bytes, size_t length)
{
    const uint8_t* p = static_cast<const uint8_t*>(bytes);
    const uint8_t* e = p + length;
    uint32_t l = *crc;
    uint64_t l64;

    // We have dedicated instruction for 1,2,4 and 8 bytes.
    if (length & 8) {
        ABSL_INTERNAL_STEP8(l, p);
        length &= ~size_t{8};
    }
    if (length & 4) {
        ABSL_INTERNAL_STEP4(l);
        length &= ~size_t{4};
    }
    if (length & 2) {
        ABSL_INTERNAL_STEP2(l);
        length &= ~size_t{2};
    }
    if (length & 1) {
        ABSL_INTERNAL_STEP1(l);
        length &= ~size_t{1};
    }
    if (length == 0) {
        *crc = l;
        return;
    }
    // length is now multiple of 16.

    // For small blocks just run simple loop, because cost of combining multiple streams is significant.
    if (length < kSmallCutoff) {
        while (length >= 16) {
            ABSL_INTERNAL_STEP8(l, p);
            ABSL_INTERNAL_STEP8(l, p);
            length -= 16;
        }
        *crc = l;
        return;
    }

    // We run 3 crc streams and combine them as described in
    // Intel paper above. Running 4th stream doesn't help, because crc
    // instruction has latency 3 and throughput 1.
    l64 = l;
    uint64_t l641 = 0;
    uint64_t l642 = 0;
    const size_t blockSize = 32;
    size_t bs = static_cast<size_t>(e - p) / kGroupsSmall / blockSize;
    const uint8_t* p1 = p + bs * blockSize;
    const uint8_t* p2 = p1 + bs * blockSize;

    for (size_t i = 0; i + 1 < bs; ++i) {
        ABSL_INTERNAL_STEP8BY3(l64, l641, l642, p, p1, p2);
        ABSL_INTERNAL_STEP8BY3(l64, l641, l642, p, p1, p2);
        ABSL_INTERNAL_STEP8BY3(l64, l641, l642, p, p1, p2);
        ABSL_INTERNAL_STEP8BY3(l64, l641, l642, p, p1, p2);
        PrefetchToLocalCache(reinterpret_cast<const char*>(p + kPrefetchHorizonMedium));
        PrefetchToLocalCache(reinterpret_cast<const char*>(p1 + kPrefetchHorizonMedium));
        PrefetchToLocalCache(reinterpret_cast<const char*>(p2 + kPrefetchHorizonMedium));
    }
    // Don't run crc on last 8 bytes.
    ABSL_INTERNAL_STEP8BY3(l64, l641, l642, p, p1, p2);
    ABSL_INTERNAL_STEP8BY3(l64, l641, l642, p, p1, p2);
    ABSL_INTERNAL_STEP8BY3(l64, l641, l642, p, p1, p2);
    ABSL_INTERNAL_STEP8BY2(l64, l641, p, p1);

    // donothing(l642);

    V128 magic = *(reinterpret_cast<const V128*>((void*)kClmulConstants) + bs - 1);

    V128 tmp = V128_From64WithZeroFill(l64);
    V128 res1 = V128_PMulLow(tmp, magic);
    tmp = V128_From64WithZeroFill(l641);
    V128 res2 = V128_PMul10(tmp, magic);
    V128 x = V128_Xor(res1, res2);
    l64 = static_cast<uint64_t>(V128_Low64(x)) ^ *(uint64_t*)(void*)(p2);
    l64 = CRC32_u64(static_cast<uint32_t>(l642), l64);

    p = p2 + 8;
    l = static_cast<uint32_t>(l64);

    while ((e - p) >= 16) {
        ABSL_INTERNAL_STEP8(l, p);
        ABSL_INTERNAL_STEP8(l, p);
    }
    // Process the last few bytes
    while (p != e) {
        ABSL_INTERNAL_STEP1(l);
    }
    * crc = l;
}

uint32_t ComputeCrc32c(std::string_view str)
{
    uint32_t crc = 0xffffffffU;
    Extend(&crc, str.data(), str.size());
    return crc ^ 0xffffffffU;
}
