#pragma once
#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <windows.h>
#include <x86intrin.h>

namespace aes_impl
{
	static inline __attribute__((always_inline)) __m128i aes_expand128(__m128i x) noexcept
	{
		__m128i t = _mm_slli_si128(x, 4);
		x = _mm_xor_si128(x, t);
		t = _mm_slli_si128(t, 4);
		x = _mm_xor_si128(x, t);
		t = _mm_slli_si128(t, 4);
		x = _mm_xor_si128(x, t);
		return x;
	}

	static inline __attribute__((always_inline)) __m128i aes_round_a(__m128i a, const __m128i &prev, uint8_t rcon) noexcept
	{
		__m128i c = _mm_aeskeygenassist_si128(prev, rcon);
		c = _mm_shuffle_epi32(c, 0xff);
		a = aes_expand128(a);
		return _mm_xor_si128(a, c);
	}

	static inline __attribute__((always_inline)) __m128i aes_round_b(__m128i b, const __m128i &a) noexcept
	{
		__m128i d = _mm_aeskeygenassist_si128(a, 0x00);
		__m128i c = _mm_shuffle_epi32(d, 0xaa);
		b = aes_expand128(b);
		return _mm_xor_si128(b, c);
	}

	static inline __attribute__((always_inline)) void aes_next_round(__m128i &a, __m128i &b, uint8_t rcon, __m128i r1[15], int &idx) noexcept
	{
		a = aes_round_a(a, b, rcon);
		r1[idx++] = a;
		b = aes_round_b(b, a);
		r1[idx++] = b;
	}

	static inline __attribute__((always_inline)) uint64_t rdrand64() noexcept
	{
		uint64_t val = 0;
		while (!_rdrand64_step(&val))
			_mm_pause();
		return val;
	}

	static inline __attribute__((always_inline)) bool is_aligned16(const void *p) noexcept
	{
		return (reinterpret_cast<uintptr_t>(p) & 0x0FUL) == 0;
	}

	static inline __attribute__((always_inline)) void prefetch_read(const void *p) noexcept
	{
		_mm_prefetch(reinterpret_cast<const char *>(p), _MM_HINT_T0);
	}

	static inline __attribute__((always_inline)) void secure_zero(void *p, size_t n) noexcept
	{
		volatile uint8_t *vp = reinterpret_cast<volatile uint8_t *>(p);
		for (size_t i = 0; i < n; ++i)
			vp[i] = 0;
	}

	static __attribute__((noinline, hot, target("aes,ssse3,sse4.1"))) void main_loop(
		const __m128i *__restrict__ r1,
		std::array<uint8_t, 16> &iv,
		const uint8_t *__restrict__ in,
		uint8_t *__restrict__ out,
		size_t n) noexcept
	{
		constexpr size_t BLOCK = 16;
		constexpr size_t LANES = 8;
		constexpr size_t CHUNK = BLOCK * LANES;
		constexpr size_t THRESHOLD = (4 << 20);

		const bool out_aligned = is_aligned16(out);

		uint64_t lo = 0;
		uint64_t hi = 0;
		std::memcpy(&lo, iv.data(), 8);
		std::memcpy(&hi, iv.data() + 8, 8);

		size_t i = 0;
		size_t limit = n / CHUNK;

		for (size_t chunk_idx = 0; chunk_idx < limit; ++chunk_idx)
		{
			__m128i s[LANES];
			uint64_t lo_copy = lo;
			uint64_t hi_copy = hi;

#pragma GCC unroll 8
			for (size_t lane = 0; lane < LANES; ++lane)
			{
				s[lane] = _mm_set_epi64x(static_cast<long long>(hi_copy), static_cast<long long>(lo_copy));
				++lo_copy;
				if (lo_copy == 0)
				{
					++hi_copy;
				}
			}

			lo = lo_copy;
			hi = hi_copy;

#pragma GCC unroll 8
			for (size_t lane = 0; lane < LANES; ++lane)
			{
				s[lane] = _mm_xor_si128(s[lane], r1[0]);
			}

#pragma GCC unroll 13
			for (int round = 1; round <= 13; ++round)
			{
				const __m128i rk = r1[round];
				for (size_t lane = 0; lane < LANES; ++lane)
				{
					s[lane] = _mm_aesenc_si128(s[lane], rk);
				}
			}

#pragma GCC unroll 8
			for (size_t lane = 0; lane < LANES; ++lane)
			{
				s[lane] = _mm_aesenclast_si128(s[lane], r1[14]);
			}

			const uint8_t *p0 = in + i;
			const uint8_t *p[LANES];

#pragma GCC unroll 8
			for (size_t lane = 0; lane < LANES; ++lane)
			{
				p[lane] = p0 + lane * BLOCK;
			}

#pragma GCC unroll 8
			for (size_t lane = 0; lane < LANES; ++lane)
			{
				prefetch_read(p[lane] + 512);
			}

			uint8_t *q0 = out + i;
			uint8_t *q[LANES];

#pragma GCC unroll 8
			for (size_t lane = 0; lane < LANES; ++lane)
			{
				q[lane] = q0 + lane * BLOCK;
			}

			__m128i tmpv[LANES];

#pragma GCC unroll 8
			for (size_t lane = 0; lane < LANES; ++lane)
			{
				tmpv[lane] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p[lane]));
			}

#pragma GCC unroll 8
			for (size_t lane = 0; lane < LANES; ++lane)
			{
				tmpv[lane] = _mm_xor_si128(tmpv[lane], s[lane]);
			}

			if (out_aligned && (n >= THRESHOLD))
			{
#pragma GCC unroll 8
				for (size_t lane = 0; lane < LANES; ++lane)
				{
					_mm_stream_si128(reinterpret_cast<__m128i *>(q[lane]), tmpv[lane]);
				}
			}
			else
			{
#pragma GCC unroll 8
				for (size_t lane = 0; lane < LANES; ++lane)
				{
					_mm_storeu_si128(reinterpret_cast<__m128i *>(q[lane]), tmpv[lane]);
				}
			}

			i += CHUNK;
		}

		for (; i + BLOCK <= n; i += BLOCK)
		{
			uint8_t ctr[16];
			std::memcpy(ctr, &lo, 8);
			std::memcpy(ctr + 8, &hi, 8);
			++lo;
			if (lo == 0)
				++hi;

			__m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ctr));
			s = _mm_xor_si128(s, r1[0]);

#pragma GCC unroll 13
			for (int r = 1; r <= 13; ++r)
			{
				s = _mm_aesenc_si128(s, r1[r]);
			}

			s = _mm_aesenclast_si128(s, r1[14]);

			__m128i ct = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
			__m128i pt = _mm_xor_si128(ct, s);

			if (out_aligned && (n >= THRESHOLD))
			{
				_mm_stream_si128(reinterpret_cast<__m128i *>(out + i), pt);
			}
			else
			{
				_mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), pt);
			}
		}

		if (i < n)
		{
			uint8_t ctr[16];
			std::memcpy(ctr, &lo, 8);
			std::memcpy(ctr + 8, &hi, 8);
			++lo;
			if (lo == 0)
			{
				++hi;
			}

			__m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ctr));
			s = _mm_xor_si128(s, r1[0]);

#pragma GCC unroll 13
			for (int r = 1; r <= 13; ++r)
			{
				s = _mm_aesenc_si128(s, r1[r]);
			}

			s = _mm_aesenclast_si128(s, r1[14]);

			alignas(16) uint8_t tmp[16];
			_mm_store_si128(reinterpret_cast<__m128i *>(tmp), s);

			size_t tail = n - i;
			for (size_t j = 0; j < tail; ++j)
			{
				out[i + j] = in[i + j] ^ tmp[j];
			}
		}

		if (out_aligned && (n >= THRESHOLD))
		{
			_mm_sfence();
		}

		std::memcpy(iv.data(), &lo, 8);
		std::memcpy(iv.data() + 8, &hi, 8);
	}

	static inline __attribute__((always_inline)) bool main_loop_file(const __m128i r1[15], HANDLE file_in, HANDLE file_out, std::array<uint8_t, 16> &iv) noexcept
	{
		constexpr size_t FILE_CHUNK = 16 * 1024 * 1024;
		std::vector<uint8_t> inbuf(FILE_CHUNK);
		std::vector<uint8_t> outbuf(FILE_CHUNK);
		DWORD bytes_read = 0;
		DWORD bytes_written = 0;

		while (ReadFile(file_in, inbuf.data(), static_cast<DWORD>(FILE_CHUNK), &bytes_read, nullptr) && bytes_read > 0)
		{
			const size_t n = static_cast<size_t>(bytes_read);
			main_loop(r1, iv, inbuf.data(), outbuf.data(), n);

			bytes_written = 0;
			if (!WriteFile(file_out, outbuf.data(), static_cast<DWORD>(n), &bytes_written, nullptr) || bytes_written != static_cast<DWORD>(n))
			{
				secure_zero(inbuf.data(), inbuf.size());
				secure_zero(outbuf.data(), outbuf.size());
				return false;
			}
		}

		secure_zero(inbuf.data(), inbuf.size());
		secure_zero(outbuf.data(), outbuf.size());
		return true;
	}
}

struct alignas(64) aes256_t
{
	__m128i r1[15];

	aes256_t(const uint8_t key[32]) noexcept
	{
		__m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(key));
		__m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(key + 16));

		r1[0] = a;
		r1[1] = b;

		constexpr uint8_t rcon[7] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40};

		int idx = 2;

#pragma GCC unroll 7
		for (int i = 0; i < 7; ++i)
		{
			aes_impl::aes_next_round(a, b, rcon[i], r1, idx);
		}
	}

	~aes256_t() noexcept
	{
		aes_impl::secure_zero(r1, sizeof(r1));
	}

	aes256_t(const aes256_t &) = delete;
	aes256_t &operator=(const aes256_t &) = delete;
	aes256_t(aes256_t &&) = delete;
	aes256_t &operator=(aes256_t &&) = delete;
};

static inline __attribute__((always_inline)) bool encrypt_bin(
	const std::vector<uint8_t> &indata,
	const std::array<uint8_t, 32> &key,
	std::vector<uint8_t> &outdata) noexcept
{
	const size_t n = indata.size();

	aes256_t aes(key.data());

	std::array<uint8_t, 16> iv{};
	uint64_t r1 = aes_impl::rdrand64();
	uint64_t r2 = aes_impl::rdrand64();
	std::memcpy(iv.data(), &r1, 8);
	std::memcpy(iv.data() + 8, &r2, 8);

	outdata.resize(16 + n);
	std::memcpy(outdata.data(), iv.data(), 16);

	const uint8_t *in = indata.data();
	uint8_t *out = outdata.data() + 16;
	aes_impl::main_loop(aes.r1, iv, in, out, n);

	return true;
}

static inline __attribute__((always_inline)) bool decrypt_bin(
	const std::vector<uint8_t> &indata,
	const std::array<uint8_t, 32> &key,
	std::vector<uint8_t> &outdata) noexcept
{
	if (indata.size() < 16)
		return false;

	const size_t n = indata.size() - 16;

	aes256_t aes(key.data());

	std::array<uint8_t, 16> iv{};
	std::memcpy(iv.data(), indata.data(), 16);
	outdata.resize(n);

	const uint8_t *in = indata.data() + 16;
	uint8_t *out = outdata.data();
	aes_impl::main_loop(aes.r1, iv, in, out, n);

	return true;
}

static __attribute__((noinline)) bool encrypt_file(
	const std::wstring &ipath,
	const std::wstring &opath,
	const std::array<uint8_t, 32> &key) noexcept
{
	HANDLE file_in = CreateFileW(ipath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
	if (file_in == INVALID_HANDLE_VALUE)
		return false;

	HANDLE file_out = CreateFileW(opath.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (file_out == INVALID_HANDLE_VALUE)
	{
		CloseHandle(file_in);
		return false;
	}

	aes256_t aes(key.data());
	std::array<uint8_t, 16> iv{};
	const uint64_t r1 = aes_impl::rdrand64();
	const uint64_t r2 = aes_impl::rdrand64();
	std::memcpy(iv.data(), &r1, 8);
	std::memcpy(iv.data() + 8, &r2, 8);

	DWORD written = 0;
	if (!WriteFile(file_out, iv.data(), 16, &written, nullptr) || written != 16)
	{
		CloseHandle(file_in);
		CloseHandle(file_out);
		return false;
	}

	const bool ok = aes_impl::main_loop_file(aes.r1, file_in, file_out, iv);
	CloseHandle(file_in);
	CloseHandle(file_out);
	return ok;
}

static __attribute__((noinline)) bool decrypt_file(
	const std::wstring &ipath,
	const std::wstring &opath,
	const std::array<uint8_t, 32> &key) noexcept
{
	HANDLE file_in = CreateFileW(ipath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
	if (file_in == INVALID_HANDLE_VALUE)
		return false;

	HANDLE file_out = CreateFileW(opath.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (file_out == INVALID_HANDLE_VALUE)
	{
		CloseHandle(file_in);
		return false;
	}

	aes256_t aes(key.data());
	std::array<uint8_t, 16> iv{};
	DWORD bytes_read = 0;
	if (!ReadFile(file_in, iv.data(), 16, &bytes_read, nullptr) || bytes_read != 16)
	{
		CloseHandle(file_in);
		CloseHandle(file_out);
		return false;
	}

	const bool ok = aes_impl::main_loop_file(aes.r1, file_in, file_out, iv);
	CloseHandle(file_in);
	CloseHandle(file_out);
	return ok;
}
