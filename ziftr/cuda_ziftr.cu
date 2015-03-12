
extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "miner.h"
}

#include <stdint.h>

// aus cpu-miner.c
extern int device_map[8];

// Speicher für Input/Output der verketteten Hashfunktionen
static uint32_t *d_hash[8];
static uint32_t *d_test[8];

extern void quark_blake512_cpu_init(int thr_id, int threads);
extern void quark_blake512_cpu_setBlock_80(void *pdata);

extern void ziftr_blake512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_outputHash, uint32_t *d_test, uint32_t table, int order);
extern void quark_groestl512_cpu_init(int thr_id, int threads);
extern void quark_groestl512_sm20_init(int thr_id, uint32_t threads);
extern void ziftr_groestl512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, uint32_t *d_test, uint32_t table, int order);
extern void ziftr_groestl512_sm20_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, uint32_t *d_test, uint32_t table, int order);
extern void quark_jh512_cpu_init(int thr_id, int threads);
extern void ziftr_jh512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, uint32_t *d_test, uint32_t table, int order);

extern void quark_skein512_cpu_init(int thr_id, int threads);
extern void ziftr_skein512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, uint32_t *d_test, uint32_t table, int order);

extern void m7_keccak512_setBlock_80(void *pdata);
extern void ziftr_keccak512_setBlock_80(void *pdata);
extern void ziftr_keccak512_cpu_hash_80(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, uint32_t *d_test,int order);
extern void ziftr_keccak512_cpu_hash_80_round2(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, uint32_t *d_test, int order);

extern void m7_keccak512_cpu_init(int thr_id, int threads);


extern void quark_check_cpu_init(int thr_id, int threads);
extern void quark_check_cpu_setTarget(const void *ptarget);
extern uint32_t quark_check_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order);

inline void zr5hash(void *state, const void *input)
{
    sph_blake512_context ctx_blake;
    sph_groestl512_context ctx_groestl;
    sph_jh512_context ctx_jh;
    sph_keccak512_context ctx_keccak;
    sph_skein512_context ctx_skein;
	uint32_t hash[16];
	static const int BLAKE = 0;
	static const int GROESTL = 1;
	static const int JH = 2;
	static const int SKEIN = 3;
	static const int arrOrder[][4] =
	{
		{ 0, 1, 2, 3 },
		{ 0, 1, 3, 2 },
		{ 0, 2, 1, 3 },
		{ 0, 2, 3, 1 },
		{ 0, 3, 1, 2 },
		{ 0, 3, 2, 1 },
		{ 1, 0, 2, 3 },
		{ 1, 0, 3, 2 },
		{ 1, 2, 0, 3 },
		{ 1, 2, 3, 0 },
		{ 1, 3, 0, 2 },
		{ 1, 3, 2, 0 },
		{ 2, 0, 1, 3 },
		{ 2, 0, 3, 1 },
		{ 2, 1, 0, 3 },
		{ 2, 1, 3, 0 },
		{ 2, 3, 0, 1 },
		{ 2, 3, 1, 0 },
		{ 3, 0, 1, 2 },
		{ 3, 0, 2, 1 },
		{ 3, 1, 0, 2 },
		{ 3, 1, 2, 0 },
		{ 3, 2, 0, 1 },
		{ 3, 2, 1, 0 }
	};

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, input, 80);
	sph_keccak512_close(&ctx_keccak, hash);
	uint32_t nOrder = hash[0] % (sizeof(arrOrder)/sizeof((arrOrder)[0]));
	int nSize = 64;
	
	for (unsigned int i = 0; i < 4; i++)
	{



		switch (arrOrder[nOrder][i])
		{
		case BLAKE:
			sph_blake512_init(&ctx_blake);
			sph_blake512(&ctx_blake, hash, nSize);
			sph_blake512_close(&ctx_blake, hash);

			break;
		case GROESTL:
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512(&ctx_groestl, hash, nSize);
			sph_groestl512_close(&ctx_groestl, hash);
			
			break;
		case JH:
			sph_jh512_init(&ctx_jh);
			sph_jh512(&ctx_jh, hash, nSize);
			sph_jh512_close(&ctx_jh, hash);
			
			break;
		case SKEIN:
			sph_skein512_init(&ctx_skein);
			sph_skein512(&ctx_skein, hash, nSize);
			sph_skein512_close(&ctx_skein, hash);
			
			break;
		default:
			break;
		}
	}
    memcpy(state, hash, 64);
}


extern bool opt_benchmark;
extern int compute_version[8];

extern "C" int scanhash_zr5(int thr_id, uint32_t *pdata,
   const  uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	const uint32_t Htarg = ptarget[7];


	const int throughput = 256 * 4096 * 2;

	static bool init[8] = {0,0,0,0,0,0,0,0};
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);

		
		cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput);
		cudaMalloc(&d_test[thr_id],      sizeof(uint32_t) * throughput);

		quark_blake512_cpu_init(thr_id, throughput);
		if (compute_version[thr_id]>30)
		quark_groestl512_cpu_init(thr_id, throughput);
        else
		quark_groestl512_sm20_init(thr_id,throughput);

		quark_jh512_cpu_init(thr_id, throughput);
		m7_keccak512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_check_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}
 
	uint32_t endiandata[20],safedata[20];
	for (int k = 0; k < 20; k++) {
		endiandata[k] = pdata[k];}


 m7_keccak512_setBlock_80((void*)endiandata);
	quark_check_cpu_setTarget(ptarget);



	do {
		int order = 0;
/// round one 
		ziftr_keccak512_cpu_hash_80(thr_id, throughput, pdata[19],d_hash[thr_id],d_test[thr_id], order++);

		ziftr_blake512_cpu_hash_64(thr_id, throughput, pdata[19],    d_hash[thr_id], d_test[thr_id], 0x00040000, order++); //0
		if (compute_version[thr_id]>30)
		ziftr_groestl512_cpu_hash_64(thr_id, throughput, pdata[19],  d_hash[thr_id], d_test[thr_id], 0x00010000, order++); //1
		else
		ziftr_groestl512_sm20_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00010000, order++); //1

		ziftr_jh512_cpu_hash_64(thr_id, throughput, pdata[19],       d_hash[thr_id], d_test[thr_id], 0x00020000, order++);  //2
		ziftr_skein512_cpu_hash_64(thr_id, throughput, pdata[19],    d_hash[thr_id], d_test[thr_id], 0x00030000, order++);  //3

		ziftr_blake512_cpu_hash_64(thr_id, throughput, pdata[19],    d_hash[thr_id], d_test[thr_id], 0x00040001, order++); //0
		if (compute_version[thr_id]>30)
		ziftr_groestl512_cpu_hash_64(thr_id, throughput, pdata[19],  d_hash[thr_id], d_test[thr_id], 0x00010001, order++); //1
		else
		ziftr_groestl512_sm20_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00010001, order++); //1

		ziftr_jh512_cpu_hash_64(thr_id, throughput, pdata[19],       d_hash[thr_id], d_test[thr_id], 0x00020001, order++);  //2
		ziftr_skein512_cpu_hash_64(thr_id, throughput, pdata[19],    d_hash[thr_id], d_test[thr_id], 0x00030001, order++);  //3

		ziftr_blake512_cpu_hash_64(thr_id, throughput, pdata[19],    d_hash[thr_id], d_test[thr_id], 0x00040002, order++); //0
		if (compute_version[thr_id]>30)
		ziftr_groestl512_cpu_hash_64(thr_id, throughput, pdata[19],  d_hash[thr_id], d_test[thr_id], 0x00010002, order++); //1
		else
		ziftr_groestl512_sm20_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00010002, order++); //1

		ziftr_jh512_cpu_hash_64(thr_id, throughput, pdata[19],       d_hash[thr_id], d_test[thr_id], 0x00020002, order++);  //2
		ziftr_skein512_cpu_hash_64(thr_id, throughput, pdata[19],    d_hash[thr_id], d_test[thr_id], 0x00030002, order++);  //3

		ziftr_blake512_cpu_hash_64(thr_id, throughput, pdata[19],    d_hash[thr_id], d_test[thr_id], 0x00040003, order++); //0
		if (compute_version[thr_id]>30)
		ziftr_groestl512_cpu_hash_64(thr_id, throughput, pdata[19],  d_hash[thr_id], d_test[thr_id], 0x00010003, order++); //1
		else
		ziftr_groestl512_sm20_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00010003, order++); //1

		ziftr_jh512_cpu_hash_64(thr_id, throughput, pdata[19],       d_hash[thr_id], d_test[thr_id], 0x00020003, order++);  //2
		ziftr_skein512_cpu_hash_64(thr_id, throughput, pdata[19],    d_hash[thr_id], d_test[thr_id], 0x00030003, order++);  //3


		/// round two 
		ziftr_keccak512_cpu_hash_80_round2(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], order++);

		ziftr_blake512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00040000, order++); //0
		if (compute_version[thr_id]>30)
		ziftr_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00010000, order++); //1
		else
		ziftr_groestl512_sm20_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00010000, order++); //1

		ziftr_jh512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00020000, order++);  //2
		ziftr_skein512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00030000, order++);  //3

		ziftr_blake512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00040001, order++); //0
		if (compute_version[thr_id]>30)
		ziftr_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00010001, order++); //1
		else
		ziftr_groestl512_sm20_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00010001, order++); //1

		ziftr_jh512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00020001, order++);  //2
		ziftr_skein512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00030001, order++);  //3

		ziftr_blake512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00040002, order++); //0
		if (compute_version[thr_id]>30)
		ziftr_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00010002, order++); //1
		else
		ziftr_groestl512_sm20_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00010002, order++); //1

		ziftr_jh512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00020002, order++);  //2
		ziftr_skein512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00030002, order++);  //3

		ziftr_blake512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00040003, order++); //0
		if (compute_version[thr_id]>30)
		ziftr_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00010003, order++); //1
        else 
		ziftr_groestl512_sm20_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00010003, order++); //1

		ziftr_jh512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00020003, order++);  //2
		ziftr_skein512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], d_test[thr_id], 0x00030003, order++);  //3



 
		uint32_t foundNonce = quark_check_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
// 		foundNonce = 10+pdata[19];
		if  (foundNonce != 0xffffffff)
		{
			uint32_t hash1[16];
			endiandata[0] = pdata[0] & (~0xFFFF0000);
			endiandata[19] = foundNonce;
			zr5hash(hash1, endiandata);
			endiandata[0] = endiandata[0] | (0xFFFF0000 & (hash1[0] & 0xFFFF0000));
			zr5hash(hash1, endiandata);
			
			if (fulltest(hash1, ptarget)) {
				pdata[19] = foundNonce;
				pdata[0] = endiandata[0]; // need to export both nonce and pok value
				*hashes_done = foundNonce - first_nonce + 1;
				return 1;
             }
		}

		if (((uint64_t)pdata[19] + (uint64_t)throughput) > (uint64_t)UINT32_MAX) {
			pdata[19]=max_nonce;
		} else {
		pdata[19] += throughput;
        }

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);
//	pdata[0] = pdata[0] & (~0xFFFF0000); // reset pok
	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
