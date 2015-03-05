#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>

#include "cuda_helper.h"

#define SPH_C64(x)    ((uint64_t)(x ## ULL))

// aus cpu-miner.c
extern "C" extern int device_map[8];
extern int compute_version[8];
// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);


#define SHL(x, n)			((x) << (n))
#define SHR(x, n)			((x) >> (n))

// Zum testen Hostcode...
/*	Hier erstmal die Tabelle mit den Konstanten für die Mix-Funktion. Kann später vll.
	mal direkt in den Code eingesetzt werden
*/

/*
 * M9_ ## s ## _ ## i  evaluates to s+i mod 9 (0 <= s <= 18, 0 <= i <= 7).
 */

#define M9_0_0    0
#define M9_0_1    1
#define M9_0_2    2
#define M9_0_3    3
#define M9_0_4    4
#define M9_0_5    5
#define M9_0_6    6
#define M9_0_7    7

#define M9_1_0    1
#define M9_1_1    2
#define M9_1_2    3
#define M9_1_3    4
#define M9_1_4    5
#define M9_1_5    6
#define M9_1_6    7
#define M9_1_7    8

#define M9_2_0    2
#define M9_2_1    3
#define M9_2_2    4
#define M9_2_3    5
#define M9_2_4    6
#define M9_2_5    7
#define M9_2_6    8
#define M9_2_7    0

#define M9_3_0    3
#define M9_3_1    4
#define M9_3_2    5
#define M9_3_3    6
#define M9_3_4    7
#define M9_3_5    8
#define M9_3_6    0
#define M9_3_7    1

#define M9_4_0    4
#define M9_4_1    5
#define M9_4_2    6
#define M9_4_3    7
#define M9_4_4    8
#define M9_4_5    0
#define M9_4_6    1
#define M9_4_7    2

#define M9_5_0    5
#define M9_5_1    6
#define M9_5_2    7
#define M9_5_3    8
#define M9_5_4    0
#define M9_5_5    1
#define M9_5_6    2
#define M9_5_7    3

#define M9_6_0    6
#define M9_6_1    7
#define M9_6_2    8
#define M9_6_3    0
#define M9_6_4    1
#define M9_6_5    2
#define M9_6_6    3
#define M9_6_7    4

#define M9_7_0    7
#define M9_7_1    8
#define M9_7_2    0
#define M9_7_3    1
#define M9_7_4    2
#define M9_7_5    3
#define M9_7_6    4
#define M9_7_7    5

#define M9_8_0    8
#define M9_8_1    0
#define M9_8_2    1
#define M9_8_3    2
#define M9_8_4    3
#define M9_8_5    4
#define M9_8_6    5
#define M9_8_7    6

#define M9_9_0    0
#define M9_9_1    1
#define M9_9_2    2
#define M9_9_3    3
#define M9_9_4    4
#define M9_9_5    5
#define M9_9_6    6
#define M9_9_7    7

#define M9_10_0   1
#define M9_10_1   2
#define M9_10_2   3
#define M9_10_3   4
#define M9_10_4   5
#define M9_10_5   6
#define M9_10_6   7
#define M9_10_7   8

#define M9_11_0   2
#define M9_11_1   3
#define M9_11_2   4
#define M9_11_3   5
#define M9_11_4   6
#define M9_11_5   7
#define M9_11_6   8
#define M9_11_7   0

#define M9_12_0   3
#define M9_12_1   4
#define M9_12_2   5
#define M9_12_3   6
#define M9_12_4   7
#define M9_12_5   8
#define M9_12_6   0
#define M9_12_7   1

#define M9_13_0   4
#define M9_13_1   5
#define M9_13_2   6
#define M9_13_3   7
#define M9_13_4   8
#define M9_13_5   0
#define M9_13_6   1
#define M9_13_7   2

#define M9_14_0   5
#define M9_14_1   6
#define M9_14_2   7
#define M9_14_3   8
#define M9_14_4   0
#define M9_14_5   1
#define M9_14_6   2
#define M9_14_7   3

#define M9_15_0   6
#define M9_15_1   7
#define M9_15_2   8
#define M9_15_3   0
#define M9_15_4   1
#define M9_15_5   2
#define M9_15_6   3
#define M9_15_7   4

#define M9_16_0   7
#define M9_16_1   8
#define M9_16_2   0
#define M9_16_3   1
#define M9_16_4   2
#define M9_16_5   3
#define M9_16_6   4
#define M9_16_7   5

#define M9_17_0   8
#define M9_17_1   0
#define M9_17_2   1
#define M9_17_3   2
#define M9_17_4   3
#define M9_17_5   4
#define M9_17_6   5
#define M9_17_7   6

#define M9_18_0   0
#define M9_18_1   1
#define M9_18_2   2
#define M9_18_3   3
#define M9_18_4   4
#define M9_18_5   5
#define M9_18_6   6
#define M9_18_7   7

/*
 * M3_ ## s ## _ ## i  evaluates to s+i mod 3 (0 <= s <= 18, 0 <= i <= 1).
 */

#define M3_0_0    0
#define M3_0_1    1
#define M3_1_0    1
#define M3_1_1    2
#define M3_2_0    2
#define M3_2_1    0
#define M3_3_0    0
#define M3_3_1    1
#define M3_4_0    1
#define M3_4_1    2
#define M3_5_0    2
#define M3_5_1    0
#define M3_6_0    0
#define M3_6_1    1
#define M3_7_0    1
#define M3_7_1    2
#define M3_8_0    2
#define M3_8_1    0
#define M3_9_0    0
#define M3_9_1    1
#define M3_10_0   1
#define M3_10_1   2
#define M3_11_0   2
#define M3_11_1   0
#define M3_12_0   0
#define M3_12_1   1
#define M3_13_0   1
#define M3_13_1   2
#define M3_14_0   2
#define M3_14_1   0
#define M3_15_0   0
#define M3_15_1   1
#define M3_16_0   1
#define M3_16_1   2
#define M3_17_0   2
#define M3_17_1   0
#define M3_18_0   0
#define M3_18_1   1

#define XCAT(x, y)     XCAT_(x, y)
#define XCAT_(x, y)    x ## y

#define SKBI(k, s, i)   XCAT(k, XCAT(XCAT(XCAT(M9_, s), _), i))
#define SKBT(t, s, v)   XCAT(t, XCAT(XCAT(XCAT(M3_, s), _), v))

#define TFBIG_KINIT(k0, k1, k2, k3, k4, k5, k6, k7, k8, t0, t1, t2) { \
		k8 = ((k0 ^ k1) ^ (k2 ^ k3)) ^ ((k4 ^ k5) ^ (k6 ^ k7)) \
			^ SPH_C64(0x1BD11BDAA9FC1A22); \
		t2 = t0 ^ t1; \
	}

#define TFBIG_ADDKEY(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
		w0 = (w0 + SKBI(k, s, 0)); \
		w1 = (w1 + SKBI(k, s, 1)); \
		w2 = (w2 + SKBI(k, s, 2)); \
		w3 = (w3 + SKBI(k, s, 3)); \
		w4 = (w4 + SKBI(k, s, 4)); \
		w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
		w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
		w7 = (w7 + SKBI(k, s, 7) + (uint64_t)s); \
	}

#define TFBIG_MIX(x0, x1, rc) { \
		x0 = x0 + x1; \
		x1 = ROTL64(x1, rc) ^ x0; \
	}

#define TFBIG_MIX8(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX(w0, w1, rc0); \
		TFBIG_MIX(w2, w3, rc1); \
		TFBIG_MIX(w4, w5, rc2); \
		TFBIG_MIX(w6, w7, rc3); \
	}

#define TFBIG_4e(s)  { \
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
		TFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
		TFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
		TFBIG_MIX8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
	}

#define TFBIG_4o(s)  { \
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
		TFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
		TFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
		TFBIG_MIX8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
	}
///////////////////////////
#define uTFBIG_KINIT(k0, k1, k2, k3, k4, k5, k6, k7, k8, t0, t1, t2) { \
		k8 = ((k0 ^ k1) ^ (k2 ^ k3)) ^ ((k4 ^ k5) ^ (k6 ^ k7)) \
			^ vectorize(0x1BD11BDAA9FC1A22); \
		t2 = t0 ^ t1; \
		}

#define uTFBIG_ADDKEY(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
		w0 = (w0 + SKBI(k, s, 0)); \
		w1 = (w1 + SKBI(k, s, 1)); \
		w2 = (w2 + SKBI(k, s, 2)); \
		w3 = (w3 + SKBI(k, s, 3)); \
		w4 = (w4 + SKBI(k, s, 4)); \
		w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
		w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
		w7 = (w7 + SKBI(k, s, 7) + vectorize(s)); \
		}

#define uTFBIG_MIX(x0, x1, rc) { \
		x0 = x0 + x1; \
		x1 = ROL2(x1, rc) ^ x0; \
		}

#define uTFBIG_MIX8(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		uTFBIG_MIX(w0, w1, rc0); \
		uTFBIG_MIX(w2, w3, rc1); \
		uTFBIG_MIX(w4, w5, rc2); \
		uTFBIG_MIX(w6, w7, rc3); \
		}

#define uTFBIG_4e(s)  { \
		uTFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		uTFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
		uTFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
		uTFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
		uTFBIG_MIX8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
		}

#define uTFBIG_4o(s)  { \
		uTFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		uTFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
		uTFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
		uTFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
		uTFBIG_MIX8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
		}
//////////////////////////////////////////
static __constant__ uint64_t d_constMem[8];
static uint64_t h_constMem[8] = {
	SPH_C64(0x4903ADFF749C51CE),
	SPH_C64(0x0D95DE399746DF03),
	SPH_C64(0x8FD1934127C79BCE),
	SPH_C64(0x9A255629FF352CB1),
	SPH_C64(0x5DB62599DF6CA7B0),
	SPH_C64(0xEABE394CA9D5C3F4),
	SPH_C64(0x991112C71A75B523),
	SPH_C64(0xAE18A40B660FCC33) };

static __constant__ uint2 t12[6] = 
{
	{ 0x40, 0x0 }, { 0, 0xf0000000 }, {0x40,0xf0000000},
	{ 0x8, 0x0 }, { 0, 0xff000000 }, {0x8,0xff000000}
};


static __device__ __forceinline__ void tfbig_addkey_uint2(uint2 &w0, uint2 &w1, uint2 &w2, uint2 &w3, uint2 &w4, uint2 &w5, uint2 &w6, uint2 &w7,
	uint2 *k, uint2 *t, int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7, int t0, int t1, int s)
{
	w0 += k[s0];
	w1 += k[s1];
	w2 += k[s2];
	w3 += k[s3];
	w4 += k[s4];
	w5 += k[s5] + t[t0];
	w6 += k[s6] + t[t1];
	w7 += k[s7] + vectorize(s);
}

static __device__ __forceinline__ void tfbig_4e_uint2(uint2 &w0, uint2 &w1, uint2 &w2, uint2 &w3, uint2 &w4, uint2 &w5, uint2 &w6, uint2 &w7)
{

	w0 += w1;
	w2 += w3;
	w4 += w5;
	w6 += w7;
	w1 = ROL2(w1, 46);
	w3 = ROL2(w3, 36);
	w5 = ROL2(w5, 19);
	w7 = ROL2(w7, 37);
	w1 ^= w0;
	w3 ^= w2;
	w5 ^= w4;
	w7 ^= w6;


	w0 += w3;
	w2 += w1;
	w4 += w7;
	w6 += w5;
	w1 = ROL2(w1, 33);
	w7 = ROL2(w7, 27);
	w5 = ROL2(w5, 14);
	w3 = ROL2(w3, 42);
	w1 ^= w2;
	w3 ^= w0;
	w5 ^= w6;
	w7 ^= w4;

	w0 += w5;
	w2 += w7;
	w4 += w1;
	w6 += w3;
	w1 = ROL2(w1, 17);
	w3 = ROL2(w3, 49);
	w5 = ROL2(w5, 36);
	w7 = ROL2(w7, 39);
	w1 ^= w4;
	w3 ^= w6;
	w5 ^= w0;
	w7 ^= w2;

	w0 += w7;
	w2 += w5;
	w4 += w3;
	w6 += w1;
	w1 = ROL2(w1, 44);
	w7 = ROL2(w7, 9);
	w5 = ROL2(w5, 54);
	w3 = ROL2(w3, 56);
	w1 ^= w6;
	w3 ^= w4;
	w5 ^= w2;
	w7 ^= w0;



}

static __device__ __forceinline__ void tfbig_4o_uint2(uint2 &w0, uint2 &w1, uint2 &w2, uint2 &w3, uint2 &w4, uint2 &w5, uint2 &w6, uint2 &w7)
{

	w0 += w1;
	w2 += w3;
	w4 += w5;
	w6 += w7;
	w1 = ROL2(w1, 39);
	w3 = ROL2(w3, 30);
	w5 = ROL2(w5, 34);
	w7 = ROL2(w7, 24);
	w1 ^= w0;
	w3 ^= w2;
	w5 ^= w4;
	w7 ^= w6;


	w0 += w3;
	w2 += w1;
	w4 += w7;
	w6 += w5;
	w1 = ROL2(w1, 13);
	w7 = ROL2(w7, 50);
	w5 = ROL2(w5, 10);
	w3 = ROL2(w3, 17);
	w1 ^= w2;
	w3 ^= w0;
	w5 ^= w6;
	w7 ^= w4;

	w0 += w5;
	w2 += w7;
	w4 += w1;
	w6 += w3;
	w1 = ROL2(w1, 25);
	w3 = ROL2(w3, 29);
	w5 = ROL2(w5, 39);
	w7 = ROL2(w7, 43);
	w1 ^= w4;
	w3 ^= w6;
	w5 ^= w0;
	w7 ^= w2;


	w0 += w7;
	w2 += w5;
	w4 += w3;
	w6 += w1;
	w1 = ROL2(w1, 8);
	w7 = ROL2(w7, 35);
	w5 = ROL2(w5, 56);
	w3 = ROL2(w3, 22);
	w1 ^= w6;
	w3 ^= w4;
	w5 ^= w2;
	w7 ^= w0;

}


static __device__ __forceinline__ void tfbig_4e(uint64_t &w0, uint64_t &w1, uint64_t &w2, uint64_t &w3, uint64_t &w4, uint64_t &w5, uint64_t &w6, uint64_t &w7)
{

	w0 += w1;
	w2 += w3;
	w4 += w5;
	w6 += w7;
	w1 = ROTL64(w1, 46);
	w3 = ROTL64(w3, 36);
	w5 = ROTL64(w5, 19);
	w7 = ROTL64(w7, 37);
	w1 = xor1(w1, w0);
	w3 = xor1(w3, w2);
	w5 = xor1(w5, w4);
	w7 = xor1(w7, w6);


	w0 += w3;
	w2 += w1;
	w4 += w7;
	w6 += w5;
	w1 = ROTL64(w1, 33);
	w7 = ROTL64(w7, 27);
	w5 = ROTL64(w5, 14);
	w3 = ROTL64(w3, 42);
	w1 = xor1(w1, w2);
	w3 = xor1(w3, w0);
	w5 = xor1(w5, w6);
	w7 = xor1(w7, w4);

	w0 += w5;
	w2 += w7;
	w4 += w1;
	w6 += w3;
	w1 = ROTL64(w1, 17);
	w3 = ROTL64(w3, 49);
	w5 = ROTL64(w5, 36);
	w7 = ROTL64(w7, 39);
	w1 = xor1(w1, w4);
	w3 = xor1(w3, w6);
	w5 = xor1(w5, w0);
	w7 = xor1(w7, w2);

	w0 += w7;
	w2 += w5;
	w4 += w3;
	w6 += w1;
	w1 = ROTL64(w1, 44);
	w7 = ROTL64(w7, 9);
	w5 = ROTL64(w5, 54);
	w3 = ROTL64(w3, 56);
	w1 = xor1(w1, w6);
	w3 = xor1(w3, w4);
	w5 = xor1(w5, w2);
	w7 = xor1(w7, w0);



}

static __device__ __forceinline__ void tfbig_addkey(uint64_t &w0, uint64_t &w1, uint64_t &w2, uint64_t &w3, uint64_t &w4, uint64_t &w5, uint64_t &w6, uint64_t &w7, uint64_t *k, uint64_t* t,
	int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7, int t0, int t1, int s)
{
	w0 += k[s0];
	w1 += k[s1];
	w2 += k[s2];
	w3 += k[s3];
	w4 += k[s4];
	w5 += k[s5] + t[t0];
	w6 += k[s6] + t[t1];
	w7 += k[s7] + s;
}


static __device__ __forceinline__ void tfbig_4o(uint64_t &w0, uint64_t &w1, uint64_t &w2, uint64_t &w3, uint64_t &w4, uint64_t &w5, uint64_t &w6, uint64_t &w7)
{

	w0 += w1;
	w2 += w3;
	w4 += w5;
	w6 += w7;
	w1 = ROTL64(w1, 39);
	w3 = ROTL64(w3, 30);
	w5 = ROTL64(w5, 34);
	w7 = ROTL64(w7, 24);
	w1 = xor1(w1, w0);
	w3 = xor1(w3, w2);
	w5 = xor1(w5, w4);
	w7 = xor1(w7, w6);

	w0 += w3;
	w2 += w1;
	w4 += w7;
	w6 += w5;
	w1 = ROTL64(w1, 13);
	w7 = ROTL64(w7, 50);
	w5 = ROTL64(w5, 10);
	w3 = ROTL64(w3, 17);
	w1 = xor1(w1, w2);
	w3 = xor1(w3, w0);
	w5 = xor1(w5, w6);
	w7 = xor1(w7, w4);

	w0 += w5;
	w2 += w7;
	w4 += w1;
	w6 += w3;
	w1 = ROTL64(w1, 25);
	w3 = ROTL64(w3, 29);
	w5 = ROTL64(w5, 39);
	w7 = ROTL64(w7, 43);
	w1 = xor1(w1, w4);
	w3 = xor1(w3, w6);
	w5 = xor1(w5, w0);
	w7 = xor1(w7, w2);


	w0 += w7;
	w2 += w5;
	w4 += w3;
	w6 += w1;
	w1 = ROTL64(w1, 8);
	w7 = ROTL64(w7, 35);
	w5 = ROTL64(w5, 56);
	w3 = ROTL64(w3, 22);
	w1 = xor1(w1, w6);
	w3 = xor1(w3, w4);
	w5 = xor1(w5, w2);
	w7 = xor1(w7, w0);

}




__global__ void quark_skein512_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// Skein
		uint64_t p[8];
		uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint64_t t0, t1, t2;

		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint64_t *inpHash = &g_hash[8 * hashPosition];

		// Initialisierung
		h0 = d_constMem[0];
		h1 = d_constMem[1];
		h2 = d_constMem[2];
		h3 = d_constMem[3];
		h4 = d_constMem[4];
		h5 = d_constMem[5];
		h6 = d_constMem[6];
		h7 = d_constMem[7];

		// 1. Runde -> etype = 480, ptr = 64, bcount = 0, data = msg		
#pragma unroll 8
		for(int i=0;i<8;i++)
			p[i] = inpHash[i];

		t0 = 64; // ptr
		t1 = 480ull << 55; // etype
		TFBIG_KINIT(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
		TFBIG_4e(0);
		TFBIG_4o(1);
		TFBIG_4e(2);
		TFBIG_4o(3);
		TFBIG_4e(4);
		TFBIG_4o(5);
		TFBIG_4e(6);
		TFBIG_4o(7);
		TFBIG_4e(8);
		TFBIG_4o(9);
		TFBIG_4e(10);
		TFBIG_4o(11);
		TFBIG_4e(12);
		TFBIG_4o(13);
		TFBIG_4e(14);
		TFBIG_4o(15);
		TFBIG_4e(16);
		TFBIG_4o(17);
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		h0 = inpHash[0] ^ p[0];
		h1 = inpHash[1] ^ p[1];
		h2 = inpHash[2] ^ p[2];
		h3 = inpHash[3] ^ p[3];
		h4 = inpHash[4] ^ p[4];
		h5 = inpHash[5] ^ p[5];
		h6 = inpHash[6] ^ p[6];
		h7 = inpHash[7] ^ p[7];

		// 2. Runde -> etype = 510, ptr = 8, bcount = 0, data = 0
#pragma unroll 8
		for(int i=0;i<8;i++)
			p[i] = 0;

		t0 = 8; // ptr
		t1 = 510ull << 55; // etype
		TFBIG_KINIT(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
		TFBIG_4e(0);
		TFBIG_4o(1);
		TFBIG_4e(2);
		TFBIG_4o(3);
		TFBIG_4e(4);
		TFBIG_4o(5);
		TFBIG_4e(6);
		TFBIG_4o(7);
		TFBIG_4e(8);
		TFBIG_4o(9);
		TFBIG_4e(10);
		TFBIG_4o(11);
		TFBIG_4e(12);
		TFBIG_4o(13);
		TFBIG_4e(14);
		TFBIG_4o(15);
		TFBIG_4e(16);
		TFBIG_4o(17);
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		// fertig
		uint64_t *outpHash = &g_hash[8 * hashPosition];

#pragma unroll 8
		for(int i=0;i<8;i++)
			outpHash[i] = p[i];
	}
}


__global__ void ziftr_skein512_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint8_t *d_test,uint32_t table)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{

			if ((d_test + 4 * thread)[table & (~0xFFFF0000)] == ((table & (~0x0000FFFF)) >> 16)) {


		// Skein
		uint64_t p[8];
		uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint64_t t0, t1, t2;

		uint32_t nounce = startNounce + thread;

		int hashPosition = nounce - startNounce;
		uint64_t *inpHash = &g_hash[8 * hashPosition];
		
		// Initialisierung
		h0 = d_constMem[0];
		h1 = d_constMem[1];
		h2 = d_constMem[2];
		h3 = d_constMem[3];
		h4 = d_constMem[4];
		h5 = d_constMem[5];
		h6 = d_constMem[6];
		h7 = d_constMem[7];

		// 1. Runde -> etype = 480, ptr = 64, bcount = 0, data = msg		
#pragma unroll 8
		for (int i = 0; i<8; i++)
			p[i] = inpHash[i];

		t0 = 64; // ptr
		t1 = 480ull << 55; // etype
		TFBIG_KINIT(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
		TFBIG_4e(0);
		TFBIG_4o(1);
		TFBIG_4e(2);
		TFBIG_4o(3);
		TFBIG_4e(4);
		TFBIG_4o(5);
		TFBIG_4e(6);
		TFBIG_4o(7);
		TFBIG_4e(8);
		TFBIG_4o(9);
		TFBIG_4e(10);
		TFBIG_4o(11);
		TFBIG_4e(12);
		TFBIG_4o(13);
		TFBIG_4e(14);
		TFBIG_4o(15);
		TFBIG_4e(16);
		TFBIG_4o(17);
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		h0 = inpHash[0] ^ p[0];
		h1 = inpHash[1] ^ p[1];
		h2 = inpHash[2] ^ p[2];
		h3 = inpHash[3] ^ p[3];
		h4 = inpHash[4] ^ p[4];
		h5 = inpHash[5] ^ p[5];
		h6 = inpHash[6] ^ p[6];
		h7 = inpHash[7] ^ p[7];

		// 2. Runde -> etype = 510, ptr = 8, bcount = 0, data = 0
#pragma unroll 8
		for (int i = 0; i<8; i++)
			p[i] = 0;

		t0 = 8; // ptr
		t1 = 510ull << 55; // etype
		TFBIG_KINIT(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
		TFBIG_4e(0);
		TFBIG_4o(1);
		TFBIG_4e(2);
		TFBIG_4o(3);
		TFBIG_4e(4);
		TFBIG_4o(5);
		TFBIG_4e(6);
		TFBIG_4o(7);
		TFBIG_4e(8);
		TFBIG_4o(9);
		TFBIG_4e(10);
		TFBIG_4o(11);
		TFBIG_4e(12);
		TFBIG_4o(13);
		TFBIG_4e(14);
		TFBIG_4o(15);
		TFBIG_4e(16);
		TFBIG_4o(17);
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		// fertig
		uint64_t *outpHash = &g_hash[8 * hashPosition];

#pragma unroll 8
		for (int i = 0; i<8; i++)
			outpHash[i] = p[i];

		
       } // table
	} // thread
}

__global__ void ziftr_skein512uint2_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint8_t *d_test, uint32_t table)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{

		if ((d_test + 4 * thread)[table & (~0xFFFF0000)] == ((table & (~0x0000FFFF)) >> 16)) {


			// Skein
			uint2 p[8];
			uint2 h0, h1, h2, h3, h4, h5, h6, h7, h8;
			uint2 t0, t1, t2;

			uint32_t nounce = startNounce + thread;

			int hashPosition = nounce - startNounce;
			uint64_t *inpHash = &g_hash[8 * hashPosition];

			// Initialisierung
			h0 = vectorize(d_constMem[0]);
			h1 = vectorize(d_constMem[1]);
			h2 = vectorize(d_constMem[2]);
			h3 = vectorize(d_constMem[3]);
			h4 = vectorize(d_constMem[4]);
			h5 = vectorize(d_constMem[5]);
			h6 = vectorize(d_constMem[6]);
			h7 = vectorize(d_constMem[7]);

			// 1. Runde -> etype = 480, ptr = 64, bcount = 0, data = msg		
#pragma unroll 8
			for (int i = 0; i<8; i++)
				p[i] = vectorize(inpHash[i]);

			t0 = vectorize(64); // ptr
			t1 = vectorize(480ull << 55); // etype
			uTFBIG_KINIT(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
			uTFBIG_4e(0);
			uTFBIG_4o(1);
			uTFBIG_4e(2);
			uTFBIG_4o(3);
			uTFBIG_4e(4);
			uTFBIG_4o(5);
			uTFBIG_4e(6);
			uTFBIG_4o(7);
			uTFBIG_4e(8);
			uTFBIG_4o(9);
			uTFBIG_4e(10);
			uTFBIG_4o(11);
			uTFBIG_4e(12);
			uTFBIG_4o(13);
			uTFBIG_4e(14);
			uTFBIG_4o(15);
			uTFBIG_4e(16);
			uTFBIG_4o(17);
			uTFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

			h0 = vectorize(inpHash[0]) ^ p[0];
			h1 = vectorize(inpHash[1]) ^ p[1];
			h2 = vectorize(inpHash[2]) ^ p[2];
			h3 = vectorize(inpHash[3]) ^ p[3];
			h4 = vectorize(inpHash[4]) ^ p[4];
			h5 = vectorize(inpHash[5]) ^ p[5];
			h6 = vectorize(inpHash[6]) ^ p[6];
			h7 = vectorize(inpHash[7]) ^ p[7];

			// 2. Runde -> etype = 510, ptr = 8, bcount = 0, data = 0
#pragma unroll 8
			for (int i = 0; i<8; i++)
				p[i] = make_uint2(0,0);

			t0 = vectorize(8); // ptr
			t1 = vectorize(510ull << 55); // etype
			uTFBIG_KINIT(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
			uTFBIG_4e(0);
			uTFBIG_4o(1);
			uTFBIG_4e(2);
			uTFBIG_4o(3);
			uTFBIG_4e(4);
			uTFBIG_4o(5);
			uTFBIG_4e(6);
			uTFBIG_4o(7);
			uTFBIG_4e(8);
			uTFBIG_4o(9);
			uTFBIG_4e(10);
			uTFBIG_4o(11);
			uTFBIG_4e(12);
			uTFBIG_4o(13);
			uTFBIG_4e(14);
			uTFBIG_4o(15);
			uTFBIG_4e(16);
			uTFBIG_4o(17);
			uTFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

			// fertig
			uint64_t *outpHash = &g_hash[8 * hashPosition];

#pragma unroll 8
			for (int i = 0; i<8; i++)
				outpHash[i] = devectorize(p[i]);


		} // table
	} // thread
}

// Setup-Funktionen
__host__ void quark_skein512_cpu_init(int thr_id, int threads)
{
	// nix zu tun ;-)
	cudaMemcpyToSymbol( d_constMem,
                        h_constMem,
                        sizeof(h_constMem),
                        0, cudaMemcpyHostToDevice);
}

__host__ void quark_skein512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const int threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	// Größe des dynamischen Shared Memory Bereichs
	size_t shared_size = 0;

	quark_skein512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

	// Strategisches Sleep Kommando zur Senkung der CPU Last
	MyStreamSynchronize(NULL, order, thr_id);
}

__host__ void ziftr_skein512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, uint32_t* d_test,uint32_t table,int order)
{
	const int threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	// Größe des dynamischen Shared Memory Bereichs
	size_t shared_size = 0;
	if (compute_version[thr_id]<50) {
	ziftr_skein512_gpu_hash_64 << <grid, block, shared_size >> >(threads, startNounce, (uint64_t*)d_hash, (uint8_t*)d_test, table);
	} else {
	ziftr_skein512uint2_gpu_hash_64 << <grid, block, shared_size >> >(threads, startNounce, (uint64_t*)d_hash, (uint8_t*)d_test,table);
    }
	// Strategisches Sleep Kommando zur Senkung der CPU Last
	MyStreamSynchronize(NULL, order, thr_id);
}
