#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <algorithm>
#include <immintrin.h>
#include <malloc.h>
#include <string>
#include <iomanip>
#include <sstream>
#include "aes256.h"

static inline double now_seconds() noexcept
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    static double inv_freq = []() noexcept {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return 1.0 / double(f.QuadPart);
        }();
    return double(t.QuadPart) * inv_freq;
}

static void pin_thread() noexcept
{
    SetThreadAffinityMask(GetCurrentThread(), 1ull);
}

static uint8_t* alloc_aligned(size_t size, size_t align = 64) noexcept
{
    if (size == 0) return nullptr;
    return reinterpret_cast<uint8_t*>(_aligned_malloc(size, align));
}

static void free_aligned(uint8_t* p) noexcept
{
    if (p) _aligned_free(p);
}

static inline void fill_random_avx2(uint8_t* buf, size_t size) noexcept
{
    const size_t STEP = 32;
    size_t i = 0;
    for (; i + STEP <= size; i += STEP)
    {
        uint64_t v0 = aes_impl::rdrand64();
        __m256i x = _mm256_set1_epi64x(static_cast<long long>(v0));
        _mm256_store_si256(reinterpret_cast<__m256i*>(buf + i), x);
    }
    if (i < size)
    {
        size_t remaining = size - i;
        size_t k = 0;
        while (k < remaining)
        {
            uint64_t v = aes_impl::rdrand64();
            size_t copy_bytes = (remaining - k) >= 8 ? 8 : (remaining - k);
            std::memcpy(buf + i + k, &v, copy_bytes);
            k += copy_bytes;
        }
    }
}

static void stats_print(const char* name, const std::vector<double>& v)
{
    if (v.empty()) return;
    double mn = *std::min_element(v.begin(), v.end());
    double mx = *std::max_element(v.begin(), v.end());
    double sm = 0.0;
    for (double x : v) sm += x;
    double avg = sm / double(v.size());
    std::printf("\n[%s RESULTS]\n", name);
    std::printf("Min : %.2f MB/s\n", mn);
    std::printf("Max : %.2f MB/s\n", mx);
    std::printf("Avg : %.2f MB/s\n", avg);
}

static void hexdump_context(const uint8_t* data, size_t pos, size_t len, size_t context = 8)
{
    size_t start = (pos >= context) ? (pos - context) : 0;
    size_t end = std::min(pos + context + 1, len);
    std::printf("Context (offsets 0x%zx..0x%zx):\n", start, end - 1);
    for (size_t i = start; i < end; ++i)
    {
        if (i == pos) std::printf(">>");
        else std::printf("  ");
        std::printf("0x%06zx: %02X\n", i, data[i]);
    }
}

static bool compare_vectors_precise(
    const std::vector<uint8_t>& expected,
    const std::vector<uint8_t>& actual
)
{
    if (expected.size() != actual.size())
    {
        std::printf("SIZE MISMATCH: expected %zu bytes, actual %zu bytes\n", expected.size(), actual.size());
        size_t minlen = std::min(expected.size(), actual.size());
        for (size_t i = 0; i < minlen; ++i)
        {
            if (expected[i] != actual[i])
            {
                std::printf("First mismatch at index 0x%zx (decimal %zu)\n", i, i);
                std::printf("expected: 0x%02X, actual: 0x%02X\n", expected[i], actual[i]);
                hexdump_context(expected.data(), i, expected.size());
                return false;
            }
        }
        std::printf("No differing byte in the shared range; truncated or extra tail present.\n");
        return false;
    }

    const size_t n = expected.size();
    for (size_t i = 0; i < n; ++i)
    {
        if (expected[i] != actual[i])
        {
            std::printf("First mismatch at index 0x%zx (decimal %zu)\n", i, i);
            std::printf("expected: 0x%02X, actual: 0x%02X\n", expected[i], actual[i]);
            hexdump_context(expected.data(), i, n);
            return false;
        }
    }
    return true;
}

static bool compare_files_precise(const std::wstring& a_path, const std::wstring& b_path)
{
    HANDLE fa = CreateFileW(a_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (fa == INVALID_HANDLE_VALUE) { std::printf("Failed to open %ls\n", a_path.c_str()); return false; }
    HANDLE fb = CreateFileW(b_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (fb == INVALID_HANDLE_VALUE) { std::printf("Failed to open %ls\n", b_path.c_str()); CloseHandle(fa); return false; }

    const size_t BUFSZ = 1 << 20;
    std::vector<uint8_t> A(BUFSZ), B(BUFSZ);
    DWORD ra = 0, rb = 0;
    size_t offset = 0;

    while (true)
    {
        BOOL oka = ReadFile(fa, A.data(), static_cast<DWORD>(BUFSZ), &ra, nullptr);
        BOOL okb = ReadFile(fb, B.data(), static_cast<DWORD>(BUFSZ), &rb, nullptr);
        if (!oka || !okb)
        {
            std::printf("ReadFile failed during comparison\n");
            CloseHandle(fa); CloseHandle(fb);
            return false;
        }
        if (ra == 0 && rb == 0) break;
        if (ra != rb)
        {
            std::printf("File size mismatch at offset 0x%zx (block): read %u vs %u bytes\n", offset, ra, rb);
            CloseHandle(fa); CloseHandle(fb);
            return false;
        }
        for (DWORD i = 0; i < ra; ++i)
        {
            if (A[i] != B[i])
            {
                size_t pos = offset + i;
                std::printf("First file mismatch at index 0x%zx (decimal %zu)\n", pos, pos);
                std::printf("expected: 0x%02X, actual: 0x%02X\n", A[i], B[i]);
                hexdump_context(A.data(), i, ra);
                CloseHandle(fa); CloseHandle(fb);
                return false;
            }
        }
        offset += ra;
    }

    CloseHandle(fa); CloseHandle(fb);
    return true;
}

static void warmup_mainloop(aes256_t& aes, uint8_t* in, uint8_t* out, size_t size)
{
    std::array<uint8_t, 16> iv{};
    uint64_t r1 = aes_impl::rdrand64();
    uint64_t r2 = aes_impl::rdrand64();
    std::memcpy(iv.data(), &r1, 8);
    std::memcpy(iv.data() + 8, &r2, 8);

    for (int i = 0; i < 2; ++i)
        aes_impl::main_loop(aes.r1, iv, in, out, size);
}

static void warmup_encrypt_bin(const std::vector<uint8_t>& in_vec,
    const std::array<uint8_t, 32>& key)
{
    std::vector<uint8_t> tmp;
    tmp.reserve(in_vec.size() + 128);

    for (int i = 0; i < 2; ++i)
    {
        tmp.clear();
        encrypt_bin(in_vec, key, tmp);
    }
}

static void warmup_decrypt_bin(const std::vector<uint8_t>& enc,
    const std::array<uint8_t, 32>& key)
{
    std::vector<uint8_t> tmp;
    tmp.reserve(enc.size());

    for (int i = 0; i < 2; ++i)
    {
        tmp.clear();
        decrypt_bin(enc, key, tmp);
    }
}

static void warmup_encrypt_file(const wchar_t* in, const wchar_t* out,
    const std::array<uint8_t, 32>& key)
{
    for (int i = 0; i < 2; ++i)
        encrypt_file(in, out, key);
}

static void warmup_decrypt_file(const wchar_t* in, const wchar_t* out,
    const std::array<uint8_t, 32>& key)
{
    for (int i = 0; i < 2; ++i)
        decrypt_file(in, out, key);
}

static bool verify_bin_roundtrip(const std::vector<uint8_t>& plain, const std::array<uint8_t, 32>& key)
{
    std::vector<uint8_t> enc;
    std::vector<uint8_t> dec;
    encrypt_bin(plain, key, enc);
    if (enc.size() < 16) { std::printf("encrypt_bin produced data smaller than IV length\n"); return false; }
    if (!decrypt_bin(enc, key, dec))
    {
        std::printf("decrypt_bin returned false\n");
        return false;
    }
    if (!compare_vectors_precise(plain, dec))
    {
        std::printf("Binary roundtrip verification FAILED\n");
        return false;
    }
    return true;
}

static bool verify_file_roundtrip(const std::wstring& ipath, const std::wstring& encpath, const std::wstring& decpath, const std::array<uint8_t, 32>& key)
{
    if (!encrypt_file(ipath, encpath, key))
    {
        std::printf("encrypt_file failed for %ls -> %ls\n", ipath.c_str(), encpath.c_str());
        return false;
    }
    if (!decrypt_file(encpath, decpath, key))
    {
        std::printf("decrypt_file failed for %ls -> %ls\n", encpath.c_str(), decpath.c_str());
        return false;
    }
    if (!compare_files_precise(ipath, decpath))
    {
        std::printf("File roundtrip verification FAILED for %ls\n", ipath.c_str());
        return false;
    }
    return true;
}

int main()
{
    std::printf("Starting AES benchmark with pre-checks...\n");
    pin_thread();

    constexpr int ITER = 10;
    constexpr size_t DATA_MB = 64;
    constexpr size_t DATA_SIZE = DATA_MB * 1024ULL * 1024ULL;

    uint8_t* in = alloc_aligned(DATA_SIZE, 64);
    uint8_t* out = alloc_aligned(DATA_SIZE + 64, 64);
    if (!in || !out)
    {
        std::printf("allocation failed\n");
        free_aligned(in);
        free_aligned(out);
        return 1;
    }

    fill_random_avx2(in, DATA_SIZE);

    std::array<uint8_t, 32> key{};
    for (int i = 0; i < 32; ++i)
        key[i] = static_cast<uint8_t>(i * 3);

    aes256_t aes(key.data());

    std::printf("Performing correctness verification (binary and file) before benchmarks...\n");

    std::vector<size_t> test_sizes = { 0, 1, 15, 16, 17, 31, 32, 1024, 4096, 65535, DATA_SIZE };
    for (size_t s : test_sizes)
    {
        std::vector<uint8_t> plain;
        plain.resize(s);
        if (s > 0)
        {
            size_t i = 0;
            while (i < s)
            {
                uint64_t r = aes_impl::rdrand64();
                size_t copy = std::min<size_t>(8, s - i);
                std::memcpy(plain.data() + i, &r, copy);
                i += copy;
            }
        }

        std::printf(" - Binary roundtrip verify: size %zu bytes ... ", s);
        bool ok = verify_bin_roundtrip(plain, key);
        if (!ok)
        {
            std::printf("FAILED\n");
            free_aligned(in);
            free_aligned(out);
            return 2;
        }
        std::printf("OK\n");
    }

    const wchar_t* bench_in = L"bench_verify_in.bin";
    const wchar_t* bench_enc = L"bench_verify_enc.bin";
    const wchar_t* bench_dec = L"bench_verify_dec.bin";

    {
        HANDLE hf = CreateFileW(bench_in, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (hf == INVALID_HANDLE_VALUE)
        {
            std::printf("Failed to create %ls for file verification\n", bench_in);
            free_aligned(in);
            free_aligned(out);
            return 3;
        }

        DWORD written = 0;
        size_t total_write = 0;
        size_t chunks[] = { 1, 15, 16, 17, 4096, 65536, 1024 * 1024 };
        for (size_t c : chunks)
        {
            std::vector<uint8_t> tmp(c);
            fill_random_avx2(tmp.data(), c);
            if (!WriteFile(hf, tmp.data(), static_cast<DWORD>(c), &written, nullptr) || written != static_cast<DWORD>(c))
            {
                std::printf("Failed writing to %ls\n", bench_in);
                CloseHandle(hf);
                free_aligned(in);
                free_aligned(out);
                return 4;
            }
            total_write += c;
        }

        size_t large = 1024 * 1024;
        std::vector<uint8_t> big(large);
        fill_random_avx2(big.data(), large);
        if (!WriteFile(hf, big.data(), static_cast<DWORD>(large), &written, nullptr) || written != static_cast<DWORD>(large))
        {
            std::printf("Failed writing large block to %ls\n", bench_in);
            CloseHandle(hf);
            free_aligned(in);
            free_aligned(out);
            return 5;
        }
        total_write += large;

        CloseHandle(hf);
        std::printf(" - Wrote verification file %ls (%zu bytes)\n", bench_in, total_write);
    }

    std::printf(" - File roundtrip verify: %ls -> %ls -> %ls ... ", bench_in, bench_enc, bench_dec);
    if (!verify_file_roundtrip(bench_in, bench_enc, bench_dec, key))
    {
        std::printf("FAILED\n");
        free_aligned(in);
        free_aligned(out);
        return 6;
    }
    std::printf("OK\n");

    std::printf("All correctness checks passed. Proceeding to benchmarks.\n");

    std::vector<double> v_mainloop, v_bin_enc, v_bin_dec, v_file_enc, v_file_dec;
    v_mainloop.reserve(ITER);
    v_bin_enc.reserve(ITER);
    v_bin_dec.reserve(ITER);
    v_file_enc.reserve(ITER);
    v_file_dec.reserve(ITER);

    std::vector<uint8_t> in_vec;
    in_vec.reserve(DATA_SIZE);
    in_vec.assign(in, in + DATA_SIZE);

    std::vector<uint8_t> bin_enc;
    std::vector<uint8_t> bin_dec;
    bin_enc.reserve(DATA_SIZE + 128);
    bin_dec.reserve(DATA_SIZE + 128);

    warmup_mainloop(aes, in, out, DATA_SIZE);
    for (int it = 0; it < ITER; ++it)
    {
        std::array<uint8_t, 16> iv{};
        uint64_t r1 = aes_impl::rdrand64();
        uint64_t r2 = aes_impl::rdrand64();
        std::memcpy(iv.data(), &r1, 8);
        std::memcpy(iv.data() + 8, &r2, 8);

        double t0 = now_seconds();
        aes_impl::main_loop(aes.r1, iv, in, out, DATA_SIZE);
        double t1 = now_seconds();

        double mbps = double(DATA_SIZE) / (1024.0 * 1024.0) / (t1 - t0);
        v_mainloop.push_back(mbps);

        std::printf("main_loop iter %d: %.2f MB/s (%.6f s)\n",
            it + 1, mbps, t1 - t0);
    }

    warmup_encrypt_bin(in_vec, key);
    for (int it = 0; it < ITER; ++it)
    {
        bin_enc.clear();
        double t0 = now_seconds();
        encrypt_bin(in_vec, key, bin_enc);
        double t1 = now_seconds();

        double mbps = double(DATA_SIZE) / (1024.0 * 1024.0) / (t1 - t0);
        v_bin_enc.push_back(mbps);

        std::printf("encrypt_bin iter %d: %.2f MB/s (%.6f s)\n",
            it + 1, mbps, t1 - t0);
    }

    warmup_decrypt_bin(bin_enc, key);
    for (int it = 0; it < ITER; ++it)
    {
        bin_dec.clear();
        double t0 = now_seconds();
        decrypt_bin(bin_enc, key, bin_dec);
        double t1 = now_seconds();

        double mbps = double(DATA_SIZE) / (1024.0 * 1024.0) / (t1 - t0);
        v_bin_dec.push_back(mbps);

        std::printf("decrypt_bin iter %d: %.2f MB/s (%.6f s)\n",
            it + 1, mbps, t1 - t0);
    }

    HANDLE hf = CreateFileW(L"bench_in.bin", GENERIC_WRITE, 0, nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr);

    if (hf == INVALID_HANDLE_VALUE)
    {
        std::printf("failed to create bench_in.bin\n");
        free_aligned(in);
        free_aligned(out);
        return 7;
    }

    DWORD written = 0;
    BOOL ok = WriteFile(hf, in, static_cast<DWORD>(DATA_SIZE), &written, nullptr);
    CloseHandle(hf);

    if (!ok || written != static_cast<DWORD>(DATA_SIZE))
    {
        std::printf("failed to write bench_in.bin\n");
        free_aligned(in);
        free_aligned(out);
        return 8;
    }

    warmup_encrypt_file(L"bench_in.bin", L"bench_enc.bin", key);
    for (int it = 0; it < ITER; ++it)
    {
        double t0 = now_seconds();
        encrypt_file(L"bench_in.bin", L"bench_enc.bin", key);
        double t1 = now_seconds();

        double mbps = double(DATA_SIZE) / (1024.0 * 1024.0) / (t1 - t0);
        v_file_enc.push_back(mbps);

        std::printf("encrypt_file iter %d: %.2f MB/s (%.6f s)\n", it + 1, mbps, t1 - t0);
    }

    warmup_decrypt_file(L"bench_enc.bin", L"bench_dec.bin", key);
    for (int it = 0; it < ITER; ++it)
    {
        double t0 = now_seconds();
        decrypt_file(L"bench_enc.bin", L"bench_dec.bin", key);
        double t1 = now_seconds();

        double mbps = double(DATA_SIZE) / (1024.0 * 1024.0) / (t1 - t0);
        v_file_dec.push_back(mbps);

        std::printf("decrypt_file iter %d: %.2f MB/s (%.6f s)\n", it + 1, mbps, t1 - t0);
    }

    stats_print("main_loop", v_mainloop);
    stats_print("encrypt_bin", v_bin_enc);
    stats_print("decrypt_bin", v_bin_dec);
    stats_print("encrypt_file", v_file_enc);
    stats_print("decrypt_file", v_file_dec);

    free_aligned(in);
    free_aligned(out);

    std::printf("Benchmark completed successfully.\n");
    return 0;
}
