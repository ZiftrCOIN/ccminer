
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdint.h>
#include <memory.h>


extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);
extern int compute_version[8];

#include "cuda_helper.h"
static __constant__ uint64_t stateo[25];
static __constant__ uint64_t RC[24];
static const uint64_t cpu_RC[24] = {
    0x0000000000000001ull, 0x0000000000008082ull,
    0x800000000000808aull, 0x8000000080008000ull,
    0x000000000000808bull, 0x0000000080000001ull,
    0x8000000080008081ull, 0x8000000000008009ull,
    0x000000000000008aull, 0x0000000000000088ull,
    0x0000000080008009ull, 0x000000008000000aull,
    0x000000008000808bull, 0x800000000000008bull,
    0x8000000000008089ull, 0x8000000000008003ull,
    0x8000000000008002ull, 0x8000000000000080ull,
    0x000000000000800aull, 0x800000008000000aull,
    0x8000000080008081ull, 0x8000000000008080ull,
    0x0000000080000001ull, 0x8000000080008008ull
};

static __constant__ uchar4 arrOrder[24] =
{
	{ 4, 1, 2, 3 },
	{ 4, 1, 3, 2 },
	{ 4, 2, 1, 3 },
	{ 4, 2, 3, 1 },
	{ 4, 3, 1, 2 },
	{ 4, 3, 2, 1 },
	{ 1, 4, 2, 3 },
	{ 1, 4, 3, 2 },
	{ 1, 2, 4, 3 },
	{ 1, 2, 3, 4 },
	{ 1, 3, 4, 2 },
	{ 1, 3, 2, 4 },
	{ 2, 4, 1, 3 },
	{ 2, 4, 3, 1 },
	{ 2, 1, 4, 3 },
	{ 2, 1, 3, 4 },
	{ 2, 3, 4, 1 },
	{ 2, 3, 1, 4 },
	{ 3, 4, 1, 2 },
	{ 3, 4, 2, 1 },
	{ 3, 1, 4, 2 },
	{ 3, 1, 2, 4 },
	{ 3, 2, 4, 1 },
	{ 3, 2, 1, 4 }
};
static __device__ __forceinline__ void keccak_block(uint64_t *s, const uint64_t *keccak_round_constants) {
    size_t i;
    uint64_t t[5], u[5], v, w;

    /* absorb input */    
    
//#pragma unroll 24
    for (i = 0; i < 24; i++) {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		
        t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
        t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
        t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
        t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
        t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24]; 
		 
        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		
		uint64_t temp0,temp1,temp2,temp3,temp4;
        temp0 = ROTL64(t[0], 1);
		temp1 = ROTL64(t[1], 1);
		temp2 = ROTL64(t[2], 1);
		temp3 = ROTL64(t[3], 1);
		temp4 = ROTL64(t[4], 1);
		u[0] = xor1(t[4],temp1);
        u[1] = xor1(t[0],temp2);
        u[2] = xor1(t[1],temp3);
        u[3] = xor1(t[2],temp4);
        u[4] = xor1(t[3],temp0);
		
        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
        s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
        s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
        s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
        s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
        s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

        /* rho pi: b[..] = rotl(a[..], ..) */
        v = s[ 1];
        s[ 1] = ROTL64(s[ 6], 44);
        s[ 6] = ROTL64(s[ 9], 20);
        s[ 9] = ROTL64(s[22], 61);
        s[22] = ROTL64(s[14], 39);
        s[14] = ROTL64(s[20], 18);
        s[20] = ROTL64(s[ 2], 62);
        s[ 2] = ROTL64(s[12], 43);
        s[12] = ROTL64(s[13], 25);
        s[13] = ROTL64(s[19],  8);
        s[19] = ROTL64(s[23], 56);
        s[23] = ROTL64(s[15], 41);
        s[15] = ROTL64(s[ 4], 27);
        s[ 4] = ROTL64(s[24], 14);
        s[24] = ROTL64(s[21],  2);
        s[21] = ROTL64(s[ 8], 55);
        s[ 8] = ROTL64(s[16], 45);
        s[16] = ROTL64(s[ 5], 36);
        s[ 5] = ROTL64(s[ 3], 28);
        s[ 3] = ROTL64(s[18], 21);
        s[18] = ROTL64(s[17], 15);
        s[17] = ROTL64(s[11], 10);
        s[11] = ROTL64(s[ 7],  6);
        s[ 7] = ROTL64(s[10],  3);
        s[10] = ROTL64(    v,  1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */		

		v = s[ 0]; w = s[ 1]; 
		s[ 0] ^= (~w) & s[ 2]; 
		s[ 1] ^= (~s[ 2]) & s[ 3]; 
		s[ 2] ^= (~s[ 3]) & s[ 4]; 
		s[ 3] ^= (~s[ 4]) & v; 
		s[ 4] ^= (~v) & w;
		v = s[ 5]; w = s[ 6];
		s[ 5] ^= (~w) & s[ 7];
		s[ 6] ^= (~s[ 7]) & s[ 8];
		s[ 7] ^= (~s[ 8]) & s[ 9];
		s[ 8] ^= (~s[ 9]) & v;
		s[ 9] ^= (~v) & w;
        v = s[10]; w = s[11];
		s[10] ^= (~w) & s[12];
		s[11] ^= (~s[12]) & s[13];
		s[12] ^= (~s[13]) & s[14];
		s[13] ^= (~s[14]) & v;
		s[14] ^= (~v) & w;
        v = s[15]; w = s[16];
		s[15] ^= (~w) & s[17];
		s[16] ^= (~s[17]) & s[18];
		s[17] ^= (~s[18]) & s[19];
		s[18] ^= (~s[19]) & v;
		s[19] ^= (~v) & w;
        v = s[20]; w = s[21];
		s[20] ^= (~w) & s[22];
		s[21] ^= (~s[22]) & s[23];
		s[22] ^= (~s[23]) & s[24];
		s[23] ^= (~s[24]) & v;
        s[24] ^= (~v) & w;
		
        /* iota: a[0,0] ^= round constant */
        s[0] ^= keccak_round_constants[i];
    }
}

static __device__ __forceinline__ void keccak_blockv4(uint2 *s, const uint64_t *keccak_round_constants) {
	size_t i;
	uint2 t[5], u[5], v, w;

	//    #pragma unroll
	for (i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROL2(t[1], 1);
		u[1] = t[0] ^ ROL2(t[2], 1);
		u[2] = t[1] ^ ROL2(t[3], 1);
		u[3] = t[2] ^ ROL2(t[4], 1);
		u[4] = t[3] ^ ROL2(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1] = ROL2(s[6], 44);
		s[6] = ROL2(s[9], 20);
		s[9] = ROL2(s[22], 61);
		s[22] = ROL2(s[14], 39);
		s[14] = ROL2(s[20], 18);
		s[20] = ROL2(s[2], 62);
		s[2] = ROL2(s[12], 43);
		s[12] = ROL2(s[13], 25);
		s[13] = ROL2(s[19], 8);
		s[19] = ROL2(s[23], 56);
		s[23] = ROL2(s[15], 41);
		s[15] = ROL2(s[4], 27);
		s[4] = ROL2(s[24], 14);
		s[24] = ROL2(s[21], 2);
		s[21] = ROL2(s[8], 55);
		s[8] = ROL2(s[16], 45);
		s[16] = ROL2(s[5], 36);
		s[5] = ROL2(s[3], 28);
		s[3] = ROL2(s[18], 21);
		s[18] = ROL2(s[17], 15);
		s[17] = ROL2(s[11], 10);
		s[11] = ROL2(s[7], 6);
		s[7] = ROL2(s[10], 3);
		s[10] = ROL2(v, 1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
		v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= vectorize(keccak_round_constants[i]);
	}
}


static __device__ __forceinline__ void keccak_blockv3(uint64_t *state, const uint64_t *keccak_round_constants)
{

	uint2 Aba, Abe, Abi, Abo, Abu;
	uint2 Aga, Age, Agi, Ago, Agu;
	uint2 Aka, Ake, Aki, Ako, Aku;
	uint2 Ama, Ame, Ami, Amo, Amu;
	uint2 Asa, Ase, Asi, Aso, Asu;
	uint2 BCa, BCe, BCi, BCo, BCu;
	uint2 Da, De, Di, Do, Du;
	uint2 Eba, Ebe, Ebi, Ebo, Ebu;
	uint2 Ega, Ege, Egi, Ego, Egu;
	uint2 Eka, Eke, Eki, Eko, Eku;
	uint2 Ema, Eme, Emi, Emo, Emu;
	uint2 Esa, Ese, Esi, Eso, Esu;
	Aba = vectorize(state[0]);
	Abe = vectorize(state[1]);
	Abi = vectorize(state[2]);
	Abo = vectorize(state[3]);
	Abu = vectorize(state[4]);
	Aga = vectorize(state[5]);
	Age = vectorize(state[6]);
	Agi = vectorize(state[7]);
	Ago = vectorize(state[8]);
	Agu = vectorize(state[9]);
	Aka = vectorize(state[10]);
	Ake = vectorize(state[11]);
	Aki = vectorize(state[12]);
	Ako = vectorize(state[13]);
	Aku = vectorize(state[14]);
	Ama = vectorize(state[15]);
	Ame = vectorize(state[16]);
	Ami = vectorize(state[17]);
	Amo = vectorize(state[18]);
	Amu = vectorize(state[19]);
	Asa = vectorize(state[20]);
	Ase = vectorize(state[21]);
	Asi = vectorize(state[22]);
	Aso = vectorize(state[23]);
	Asu = vectorize(state[24]);
    #pragma unroll 
	for (int round = 0; round < 24; round += 2)
	{
		//    int round =2;
		//    prepareTheta
		BCa = Aba^Aga^Aka^Ama^Asa;
		BCe = Abe^Age^Ake^Ame^Ase;
		BCi = Abi^Agi^Aki^Ami^Asi;
		BCo = Abo^Ago^Ako^Amo^Aso;
		BCu = Abu^Agu^Aku^Amu^Asu;

		//thetaRhoPiChiIotaPrepareTheta(round  , A, E)
		Da = BCu^ROL2(BCe, 1);
		De = BCa^ROL2(BCi, 1);
		Di = BCe^ROL2(BCo, 1);
		Do = BCi^ROL2(BCu, 1);
		Du = BCo^ROL2(BCa, 1);

		Aba ^= Da;
		BCa = Aba;
		Age ^= De;
		BCe = ROL2(Age, 44);
		Aki ^= Di;
		BCi = ROL2(Aki, 43);
		Amo ^= Do;
		BCo = ROL2(Amo, 21);
		Asu ^= Du;
		BCu = ROL2(Asu, 14);
		Eba = BCa ^ ((~BCe)&  BCi);
		Eba ^= vectorize(keccak_round_constants[round]);
		Ebe = BCe ^ ((~BCi)&  BCo);
		Ebi = BCi ^ ((~BCo)&  BCu);
		Ebo = BCo ^ ((~BCu)&  BCa);
		Ebu = BCu ^ ((~BCa)&  BCe);

		Abo ^= Do;
		BCa = ROL2(Abo, 28);
		Agu ^= Du;
		BCe = ROL2(Agu, 20);
		Aka ^= Da;
		BCi = ROL2(Aka, 3);
		Ame ^= De;
		BCo = ROL2(Ame, 45);
		Asi ^= Di;
		BCu = ROL2(Asi, 61);
		Ega = BCa ^ ((~BCe)&  BCi);
		Ege = BCe ^ ((~BCi)&  BCo);
		Egi = BCi ^ ((~BCo)&  BCu);
		Ego = BCo ^ ((~BCu)&  BCa);
		Egu = BCu ^ ((~BCa)&  BCe);

		Abe ^= De;
		BCa = ROL2(Abe, 1);
		Agi ^= Di;
		BCe = ROL2(Agi, 6);
		Ako ^= Do;
		BCi = ROL2(Ako, 25);
		Amu ^= Du;
		BCo = ROL2(Amu, 8);
		Asa ^= Da;
		BCu = ROL2(Asa, 18);
		Eka = BCa ^ ((~BCe)&  BCi);
		Eke = BCe ^ ((~BCi)&  BCo);
		Eki = BCi ^ ((~BCo)&  BCu);
		Eko = BCo ^ ((~BCu)&  BCa);
		Eku = BCu ^ ((~BCa)&  BCe);

		Abu ^= Du;
		BCa = ROL2(Abu, 27);
		Aga ^= Da;
		BCe = ROL2(Aga, 36);
		Ake ^= De;
		BCi = ROL2(Ake, 10);
		Ami ^= Di;
		BCo = ROL2(Ami, 15);
		Aso ^= Do;
		BCu = ROL2(Aso, 56);
		Ema = BCa ^ ((~BCe)&  BCi);
		Eme = BCe ^ ((~BCi)&  BCo);
		Emi = BCi ^ ((~BCo)&  BCu);
		Emo = BCo ^ ((~BCu)&  BCa);
		Emu = BCu ^ ((~BCa)&  BCe);

		Abi ^= Di;
		BCa = ROL2(Abi, 62);
		Ago ^= Do;
		BCe = ROL2(Ago, 55);
		Aku ^= Du;
		BCi = ROL2(Aku, 39);
		Ama ^= Da;
		BCo = ROL2(Ama, 41);
		Ase ^= De;
		BCu = ROL2(Ase, 2);
		Esa = BCa ^ ((~BCe)&  BCi);
		Ese = BCe ^ ((~BCi)&  BCo);
		Esi = BCi ^ ((~BCo)&  BCu);
		Eso = BCo ^ ((~BCu)&  BCa);
		Esu = BCu ^ ((~BCa)&  BCe);

		//    prepareTheta
		BCa = Eba^Ega^Eka^Ema^Esa;
		BCe = Ebe^Ege^Eke^Eme^Ese;
		BCi = Ebi^Egi^Eki^Emi^Esi;
		BCo = Ebo^Ego^Eko^Emo^Eso;
		BCu = Ebu^Egu^Eku^Emu^Esu;

		//thetaRhoPiChiIotaPrepareTheta(round+1, E, A)
		Da = BCu^ROL2(BCe, 1);
		De = BCa^ROL2(BCi, 1);
		Di = BCe^ROL2(BCo, 1);
		Do = BCi^ROL2(BCu, 1);
		Du = BCo^ROL2(BCa, 1);

		Eba ^= Da;
		BCa = Eba;
		Ege ^= De;
		BCe = ROL2(Ege, 44);
		Eki ^= Di;
		BCi = ROL2(Eki, 43);
		Emo ^= Do;
		BCo = ROL2(Emo, 21);
		Esu ^= Du;
		BCu = ROL2(Esu, 14);
		Aba = BCa ^ ((~BCe)&  BCi);
		Aba ^= vectorize(keccak_round_constants[round + 1]);
		Abe = BCe ^ ((~BCi)&  BCo);
		Abi = BCi ^ ((~BCo)&  BCu);
		Abo = BCo ^ ((~BCu)&  BCa);
		Abu = BCu ^ ((~BCa)&  BCe);

		Ebo ^= Do;
		BCa = ROL2(Ebo, 28);
		Egu ^= Du;
		BCe = ROL2(Egu, 20);
		Eka ^= Da;
		BCi = ROL2(Eka, 3);
		Eme ^= De;
		BCo = ROL2(Eme, 45);
		Esi ^= Di;
		BCu = ROL2(Esi, 61);
		Aga = BCa ^ ((~BCe)&  BCi);
		Age = BCe ^ ((~BCi)&  BCo);
		Agi = BCi ^ ((~BCo)&  BCu);
		Ago = BCo ^ ((~BCu)&  BCa);
		Agu = BCu ^ ((~BCa)&  BCe);

		Ebe ^= De;
		BCa = ROL2(Ebe, 1);
		Egi ^= Di;
		BCe = ROL2(Egi, 6);
		Eko ^= Do;
		BCi = ROL2(Eko, 25);
		Emu ^= Du;
		BCo = ROL2(Emu, 8);
		Esa ^= Da;
		BCu = ROL2(Esa, 18);
		Aka = BCa ^ ((~BCe)&  BCi);
		Ake = BCe ^ ((~BCi)&  BCo);
		Aki = BCi ^ ((~BCo)&  BCu);
		Ako = BCo ^ ((~BCu)&  BCa);
		Aku = BCu ^ ((~BCa)&  BCe);

		Ebu ^= Du;
		BCa = ROL2(Ebu, 27);
		Ega ^= Da;
		BCe = ROL2(Ega, 36);
		Eke ^= De;
		BCi = ROL2(Eke, 10);
		Emi ^= Di;
		BCo = ROL2(Emi, 15);
		Eso ^= Do;
		BCu = ROL2(Eso, 56);
		Ama = BCa ^ ((~BCe)&  BCi);
		Ame = BCe ^ ((~BCi)&  BCo);
		Ami = BCi ^ ((~BCo)&  BCu);
		Amo = BCo ^ ((~BCu)&  BCa);
		Amu = BCu ^ ((~BCa)&  BCe);

		Ebi ^= Di;
		BCa = ROL2(Ebi, 62);
		Ego ^= Do;
		BCe = ROL2(Ego, 55);
		Eku ^= Du;
		BCi = ROL2(Eku, 39);
		Ema ^= Da;
		BCo = ROL2(Ema, 41);
		Ese ^= De;
		BCu = ROL2(Ese, 2);
		Asa = BCa ^ ((~BCe)&  BCi);
		Ase = BCe ^ ((~BCi)&  BCo);
		Asi = BCi ^ ((~BCo)&  BCu);
		Aso = BCo ^ ((~BCu)&  BCa);
		Asu = BCu ^ ((~BCa)&  BCe);
	}



	state[0] = devectorize(Aba);
	state[1] = devectorize(Abe);
	state[2] = devectorize(Abi);
	state[3] = devectorize(Abo);
	state[4] = devectorize(Abu);
	state[5] = devectorize(Aga);
	state[6] = devectorize(Age);
	state[7] = devectorize(Agi);
	state[8] = devectorize(Ago);
	state[9] = devectorize(Agu);
	state[10] = devectorize(Aka);
	state[11] = devectorize(Ake);
	state[12] = devectorize(Aki);
	state[13] = devectorize(Ako);
	state[14] = devectorize(Aku);
	state[15] = devectorize(Ama);
	state[16] = devectorize(Ame);
	state[17] = devectorize(Ami);
	state[18] = devectorize(Amo);
	state[19] = devectorize(Amu);
	state[20] = devectorize(Asa);
	state[21] = devectorize(Ase);
	state[22] = devectorize(Asi);
	state[23] = devectorize(Aso);
	state[24] = devectorize(Asu);


	//	if (thread == 0) {for (int i=0;i<25;i++) {printf("i%d uint2 %08x %08x\n",i, LOWORD(state[i]), HIWORD(state[i])); }}
}



static __device__ __forceinline__ void keccak_blockv2(uint64_t *state, const uint64_t *keccak_round_constants) 
{



	{
		uint64_t Aba, Abe, Abi, Abo, Abu;
		uint64_t Aga, Age, Agi, Ago, Agu;
		uint64_t Aka, Ake, Aki, Ako, Aku;
		uint64_t Ama, Ame, Ami, Amo, Amu;
		uint64_t Asa, Ase, Asi, Aso, Asu;
		uint64_t BCa, BCe, BCi, BCo, BCu;
		uint64_t Da, De, Di, Do, Du;
		uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
		uint64_t Ega, Ege, Egi, Ego, Egu;
		uint64_t Eka, Eke, Eki, Eko, Eku;
		uint64_t Ema, Eme, Emi, Emo, Emu;
		uint64_t Esa, Ese, Esi, Eso, Esu;
#define    ROL ROTL64

		//copyFromState(A, state)
		Aba = state[0];
		Abe = state[1];
		Abi = state[2];
		Abo = state[3];
		Abu = state[4];
		Aga = state[5];
		Age = state[6];
		Agi = state[7];
		Ago = state[8];
		Agu = state[9];
		Aka = state[10];
		Ake = state[11];
		Aki = state[12];
		Ako = state[13];
		Aku = state[14];
		Ama = state[15];
		Ame = state[16];
		Ami = state[17];
		Amo = state[18];
		Amu = state[19];
		Asa = state[20];
		Ase = state[21];
		Asi = state[22];
		Aso = state[23];
		Asu = state[24];

		for (int round = 0; round < 24; round += 2)
		{
			//    prepareTheta
			BCa = Aba^Aga^Aka^Ama^Asa;
			BCe = Abe^Age^Ake^Ame^Ase;
			BCi = Abi^Agi^Aki^Ami^Asi;
			BCo = Abo^Ago^Ako^Amo^Aso;
			BCu = Abu^Agu^Aku^Amu^Asu;

			//thetaRhoPiChiIotaPrepareTheta(round  , A, E)
			Da = BCu^ROL(BCe, 1);
			De = BCa^ROL(BCi, 1);
			Di = BCe^ROL(BCo, 1);
			Do = BCi^ROL(BCu, 1);
			Du = BCo^ROL(BCa, 1);

			Aba ^= Da;
			BCa = Aba;
			Age ^= De;
			BCe = ROL(Age, 44);
			Aki ^= Di;
			BCi = ROL(Aki, 43);
			Amo ^= Do;
			BCo = ROL(Amo, 21);
			Asu ^= Du;
			BCu = ROL(Asu, 14);
			Eba = BCa ^ ((~BCe)&  BCi);
			Eba ^= keccak_round_constants[round];
			Ebe = BCe ^ ((~BCi)&  BCo);
			Ebi = BCi ^ ((~BCo)&  BCu);
			Ebo = BCo ^ ((~BCu)&  BCa);
			Ebu = BCu ^ ((~BCa)&  BCe);

			Abo ^= Do;
			BCa = ROL(Abo, 28);
			Agu ^= Du;
			BCe = ROL(Agu, 20);
			Aka ^= Da;
			BCi = ROL(Aka, 3);
			Ame ^= De;
			BCo = ROL(Ame, 45);
			Asi ^= Di;
			BCu = ROL(Asi, 61);
			Ega = BCa ^ ((~BCe)&  BCi);
			Ege = BCe ^ ((~BCi)&  BCo);
			Egi = BCi ^ ((~BCo)&  BCu);
			Ego = BCo ^ ((~BCu)&  BCa);
			Egu = BCu ^ ((~BCa)&  BCe);

			Abe ^= De;
			BCa = ROL(Abe, 1);
			Agi ^= Di;
			BCe = ROL(Agi, 6);
			Ako ^= Do;
			BCi = ROL(Ako, 25);
			Amu ^= Du;
			BCo = ROL(Amu, 8);
			Asa ^= Da;
			BCu = ROL(Asa, 18);
			Eka = BCa ^ ((~BCe)&  BCi);
			Eke = BCe ^ ((~BCi)&  BCo);
			Eki = BCi ^ ((~BCo)&  BCu);
			Eko = BCo ^ ((~BCu)&  BCa);
			Eku = BCu ^ ((~BCa)&  BCe);

			Abu ^= Du;
			BCa = ROL(Abu, 27);
			Aga ^= Da;
			BCe = ROL(Aga, 36);
			Ake ^= De;
			BCi = ROL(Ake, 10);
			Ami ^= Di;
			BCo = ROL(Ami, 15);
			Aso ^= Do;
			BCu = ROL(Aso, 56);
			Ema = BCa ^ ((~BCe)&  BCi);
			Eme = BCe ^ ((~BCi)&  BCo);
			Emi = BCi ^ ((~BCo)&  BCu);
			Emo = BCo ^ ((~BCu)&  BCa);
			Emu = BCu ^ ((~BCa)&  BCe);

			Abi ^= Di;
			BCa = ROL(Abi, 62);
			Ago ^= Do;
			BCe = ROL(Ago, 55);
			Aku ^= Du;
			BCi = ROL(Aku, 39);
			Ama ^= Da;
			BCo = ROL(Ama, 41);
			Ase ^= De;
			BCu = ROL(Ase, 2);
			Esa = BCa ^ ((~BCe)&  BCi);
			Ese = BCe ^ ((~BCi)&  BCo);
			Esi = BCi ^ ((~BCo)&  BCu);
			Eso = BCo ^ ((~BCu)&  BCa);
			Esu = BCu ^ ((~BCa)&  BCe);

			//    prepareTheta
			BCa = Eba^Ega^Eka^Ema^Esa;
			BCe = Ebe^Ege^Eke^Eme^Ese;
			BCi = Ebi^Egi^Eki^Emi^Esi;
			BCo = Ebo^Ego^Eko^Emo^Eso;
			BCu = Ebu^Egu^Eku^Emu^Esu;

			//thetaRhoPiChiIotaPrepareTheta(round+1, E, A)
			Da = BCu^ROL(BCe, 1);
			De = BCa^ROL(BCi, 1);
			Di = BCe^ROL(BCo, 1);
			Do = BCi^ROL(BCu, 1);
			Du = BCo^ROL(BCa, 1);

			Eba ^= Da;
			BCa = Eba;
			Ege ^= De;
			BCe = ROL(Ege, 44);
			Eki ^= Di;
			BCi = ROL(Eki, 43);
			Emo ^= Do;
			BCo = ROL(Emo, 21);
			Esu ^= Du;
			BCu = ROL(Esu, 14);
			Aba = BCa ^ ((~BCe)&  BCi);
			Aba ^= keccak_round_constants[round + 1];
			Abe = BCe ^ ((~BCi)&  BCo);
			Abi = BCi ^ ((~BCo)&  BCu);
			Abo = BCo ^ ((~BCu)&  BCa);
			Abu = BCu ^ ((~BCa)&  BCe);

			Ebo ^= Do;
			BCa = ROL(Ebo, 28);
			Egu ^= Du;
			BCe = ROL(Egu, 20);
			Eka ^= Da;
			BCi = ROL(Eka, 3);
			Eme ^= De;
			BCo = ROL(Eme, 45);
			Esi ^= Di;
			BCu = ROL(Esi, 61);
			Aga = BCa ^ ((~BCe)&  BCi);
			Age = BCe ^ ((~BCi)&  BCo);
			Agi = BCi ^ ((~BCo)&  BCu);
			Ago = BCo ^ ((~BCu)&  BCa);
			Agu = BCu ^ ((~BCa)&  BCe);

			Ebe ^= De;
			BCa = ROL(Ebe, 1);
			Egi ^= Di;
			BCe = ROL(Egi, 6);
			Eko ^= Do;
			BCi = ROL(Eko, 25);
			Emu ^= Du;
			BCo = ROL(Emu, 8);
			Esa ^= Da;
			BCu = ROL(Esa, 18);
			Aka = BCa ^ ((~BCe)&  BCi);
			Ake = BCe ^ ((~BCi)&  BCo);
			Aki = BCi ^ ((~BCo)&  BCu);
			Ako = BCo ^ ((~BCu)&  BCa);
			Aku = BCu ^ ((~BCa)&  BCe);

			Ebu ^= Du;
			BCa = ROL(Ebu, 27);
			Ega ^= Da;
			BCe = ROL(Ega, 36);
			Eke ^= De;
			BCi = ROL(Eke, 10);
			Emi ^= Di;
			BCo = ROL(Emi, 15);
			Eso ^= Do;
			BCu = ROL(Eso, 56);
			Ama = BCa ^ ((~BCe)&  BCi);
			Ame = BCe ^ ((~BCi)&  BCo);
			Ami = BCi ^ ((~BCo)&  BCu);
			Amo = BCo ^ ((~BCu)&  BCa);
			Amu = BCu ^ ((~BCa)&  BCe);

			Ebi ^= Di;
			BCa = ROL(Ebi, 62);
			Ego ^= Do;
			BCe = ROL(Ego, 55);
			Eku ^= Du;
			BCi = ROL(Eku, 39);
			Ema ^= Da;
			BCo = ROL(Ema, 41);
			Ese ^= De;
			BCu = ROL(Ese, 2);
			Asa = BCa ^ ((~BCe)&  BCi);
			Ase = BCe ^ ((~BCi)&  BCo);
			Asi = BCi ^ ((~BCo)&  BCu);
			Aso = BCo ^ ((~BCu)&  BCa);
			Asu = BCu ^ ((~BCa)&  BCe);
		}

		//copyToState(state, A)
		state[0] = Aba;
		state[1] = Abe;
		state[2] = Abi;
		state[3] = Abo;
		state[4] = Abu;
		state[5] = Aga;
		state[6] = Age;
		state[7] = Agi;
		state[8] = Ago;
		state[9] = Agu;
		state[10] = Aka;
		state[11] = Ake;
		state[12] = Aki;
		state[13] = Ako;
		state[14] = Aku;
		state[15] = Ama;
		state[16] = Ame;
		state[17] = Ami;
		state[18] = Amo;
		state[19] = Amu;
		state[20] = Asa;
		state[21] = Ase;
		state[22] = Asi;
		state[23] = Aso;
		state[24] = Asu;

#undef    ROL
	}
}



static __forceinline__ void keccak_block_host(uint64_t *s, const uint64_t *keccak_round_constants) {
    size_t i;
    uint64_t t[5], u[5], v, w;

    /* absorb input */    
    
    for (i = 0; i < 24; i++) {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
        t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
        t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
        t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
        t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
        t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
        u[0] = t[4] ^ ROTL64(t[1], 1);
        u[1] = t[0] ^ ROTL64(t[2], 1);
        u[2] = t[1] ^ ROTL64(t[3], 1);
        u[3] = t[2] ^ ROTL64(t[4], 1);
        u[4] = t[3] ^ ROTL64(t[0], 1);

        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
        s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
        s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
        s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
        s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
        s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

        /* rho pi: b[..] = rotl(a[..], ..) */
        v = s[ 1];
        s[ 1] = ROTL64(s[ 6], 44);
        s[ 6] = ROTL64(s[ 9], 20);
        s[ 9] = ROTL64(s[22], 61);
        s[22] = ROTL64(s[14], 39);
        s[14] = ROTL64(s[20], 18);
        s[20] = ROTL64(s[ 2], 62);
        s[ 2] = ROTL64(s[12], 43);
        s[12] = ROTL64(s[13], 25);
        s[13] = ROTL64(s[19],  8);
        s[19] = ROTL64(s[23], 56);
        s[23] = ROTL64(s[15], 41);
        s[15] = ROTL64(s[ 4], 27);
        s[ 4] = ROTL64(s[24], 14);
        s[24] = ROTL64(s[21],  2);
        s[21] = ROTL64(s[ 8], 55);
        s[ 8] = ROTL64(s[16], 45);
        s[16] = ROTL64(s[ 5], 36);
        s[ 5] = ROTL64(s[ 3], 28);
        s[ 3] = ROTL64(s[18], 21);
        s[18] = ROTL64(s[17], 15);
        s[17] = ROTL64(s[11], 10);
        s[11] = ROTL64(s[ 7],  6);
        s[ 7] = ROTL64(s[10],  3);
        s[10] = ROTL64(    v,  1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
        v = s[ 0]; w = s[ 1]; s[ 0] ^= (~w) & s[ 2]; s[ 1] ^= (~s[ 2]) & s[ 3]; s[ 2] ^= (~s[ 3]) & s[ 4]; s[ 3] ^= (~s[ 4]) & v; s[ 4] ^= (~v) & w;
        v = s[ 5]; w = s[ 6]; s[ 5] ^= (~w) & s[ 7]; s[ 6] ^= (~s[ 7]) & s[ 8]; s[ 7] ^= (~s[ 8]) & s[ 9]; s[ 8] ^= (~s[ 9]) & v; s[ 9] ^= (~v) & w;
        v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
        v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
        v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

        /* iota: a[0,0] ^= round constant */
        s[0] ^= keccak_round_constants[i];
    }
}



 __constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)



__global__ __launch_bounds__(128,4) void  m7_keccak512_gpu_hash_120(int threads, uint32_t startNounce, uint64_t *outputHash)
{

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        
		uint32_t nounce = startNounce + thread;

         uint64_t state[25];

        #pragma unroll 16
		 for (int i=9;i<25;i++) {state[i]=stateo[i];}

		state[0] = xor1(stateo[0],c_PaddedMessage80[9]);
		state[1] = xor1(stateo[1],c_PaddedMessage80[10]);
		state[2] = xor1(stateo[2],c_PaddedMessage80[11]);
		state[3] = xor1(stateo[3],c_PaddedMessage80[12]);
		state[4] = xor1(stateo[4],c_PaddedMessage80[13]);
		state[5] = xor1(stateo[5],REPLACE_HIWORD(c_PaddedMessage80[14],nounce));
		state[6] = xor1(stateo[6],c_PaddedMessage80[15]);
		state[7] = stateo[7];
		state[8] = xor1(stateo[8],0x8000000000000000);
		 
//		keccak_block(state,RC);
		keccak_blockv3(state, RC);
#pragma unroll 8 
for (int i=0;i<8;i++) {outputHash[i*threads+thread]=state[i];}


	} //thread
}

__global__ __launch_bounds__(256,3) void ziftr_keccak512_gpu_hash_80(int threads, uint32_t startNounce, uint32_t *outputHash,uint32_t *test)
{

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        
		
        uint32_t nounce = startNounce +  thread ; // original implementation
	//	uint32_t nounce = cuda_swab32(nounce2);
       
		 uint2 ustate[25];
         #pragma unroll 25;
		 for(int i=0;i<25;i++) {ustate[i]=vectorize(stateo[i]);}
		 
		 uint2 addnonce; 
        LOHI(addnonce.x,addnonce.y,c_PaddedMessage80[9]);
        addnonce.y = nounce;
		ustate[0] ^= addnonce;
		ustate[1] ^= vectorize(c_PaddedMessage80[10]);
		ustate[8] ^= make_uint2(0x0,0x80000000);
		 
		keccak_blockv4(ustate, RC);

	 

#pragma unroll 8 
		for (int i = 0; i<8; i++) 
			 ((uint64_t*)(outputHash+16*thread))[i] = devectorize(ustate[i]);
		
		(test+thread)[0] = ((uint32_t*)arrOrder)[ustate[0].x % 24];
		

	} //thread
}

__global__ __launch_bounds__(256, 3) void ziftr_keccak512_gpu_hash_80_round2(int threads, uint32_t startNounce, uint32_t *outputHash, uint32_t *test)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{


		uint32_t nounce = startNounce + thread; 

		uint2 ustate[25];

		#pragma unroll
		for (int i = 9; i<25; i++) { ustate[i] = make_uint2(0,0); }
        #pragma unroll
		for (int i = 0; i<9; i++) {  ustate[i] = vectorize(c_PaddedMessage80[i]); }
		ustate[0].x |= (0xFFFF0000 & ((outputHash + 16 * thread)[0] & 0xFFFF0000));
		
		keccak_blockv4(ustate, RC);

		uint2 addnonce;
		LOHI(addnonce.x, addnonce.y, c_PaddedMessage80[9]);
		addnonce.y = nounce;
		ustate[0] ^= addnonce;
		ustate[1] ^= vectorize(c_PaddedMessage80[10]);
		ustate[8] ^= make_uint2(0x0, 0x80000000);

		keccak_blockv4(ustate, RC);

        #pragma unroll 8 
		for (int i = 0; i<8; i++)
			((uint64_t*)(outputHash + 16 * thread))[i] = devectorize(ustate[i]);

		(test + thread)[0] = ((uint32_t*)arrOrder)[ustate[0].x % 24];
	} //thread
}

__global__ __launch_bounds__(128, 2) void ziftr_keccak512_gpu_hash_80_v30(int threads, uint32_t startNounce, uint32_t *outputHash, uint32_t *test)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{


		uint32_t nounce = startNounce + thread; // original implementation
		//	uint32_t nounce = cuda_swab32(nounce2);

		uint64_t ustate[25];
#pragma unroll 25;
		for (int i = 0; i<25; i++) { ustate[i] = stateo[i]; }

		uint2 addnonce;
		LOHI(addnonce.x, addnonce.y, c_PaddedMessage80[9]);
		addnonce.y = nounce;
		ustate[0] ^= devectorize(addnonce);
		ustate[1] ^= c_PaddedMessage80[10];
		ustate[8] ^= 0x8000000000000000;

		keccak_block(ustate, RC);



#pragma unroll 8 
		for (int i = 0; i<8; i++)
			((uint64_t*)(outputHash + 16 * thread))[i] = ustate[i];

		(test + thread)[0] = ((uint32_t*)arrOrder)[((uint32_t*)ustate)[0] % 24];


	} //thread
}

__global__ __launch_bounds__(128, 2) void ziftr_keccak512_gpu_hash_80_round2_v30(int threads, uint32_t startNounce, uint32_t *outputHash, uint32_t *test)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{


		uint32_t nounce = startNounce + thread;

		uint64_t ustate[25];

#pragma unroll
		for (int i = 9; i<25; i++) { ustate[i] = 0; }
#pragma unroll
		for (int i = 0; i<9; i++) { ustate[i] = c_PaddedMessage80[i]; }
		((uint32_t*)ustate)[0] |= (0xFFFF0000 & ((outputHash + 16 * thread)[0] & 0xFFFF0000));

		keccak_block(ustate, RC);

		uint2 addnonce;
		LOHI(addnonce.x, addnonce.y, c_PaddedMessage80[9]);
		addnonce.y = nounce;
		ustate[0] ^= devectorize(addnonce);
		ustate[1] ^= c_PaddedMessage80[10];
		ustate[8] ^=  0x8000000000000000;

		keccak_block(ustate, RC);

#pragma unroll 8 
		for (int i = 0; i<8; i++)
			((uint64_t*)(outputHash + 16 * thread))[i] = ustate[i];

		(test + thread)[0] = ((uint32_t*)arrOrder)[((uint32_t*)ustate)[0] % 24];
	} //thread
}


void m7_keccak512_cpu_init(int thr_id, int threads)
{
    	
	cudaMemcpyToSymbol( RC,cpu_RC,sizeof(cpu_RC),0,cudaMemcpyHostToDevice);	
} 

__host__ void m7_keccak512_setBlock_120(void *pdata)
{

	unsigned char PaddedMessage[128];
	uint8_t ending =0x01;
	memcpy(PaddedMessage, pdata, 122);
	memset(PaddedMessage+122,ending,1); 
	memset(PaddedMessage+123, 0, 5); 
	cudaMemcpyToSymbol( c_PaddedMessage80, PaddedMessage, 16*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	uint64_t* alt_data = (uint64_t*) pdata;
         uint64_t state[25];
		 for(int i=0;i<25;i++) {state[i]=0;}
           alt_data[0] &= (~0xFFFF0000);    //// attention modif for ziftrcoin

		for (int i=0;i<9;i++) {state[i]  ^= alt_data[i];}
		
		keccak_block_host(state,cpu_RC);

		cudaMemcpyToSymbol(stateo, state, 25*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);

}


__host__ void m7_keccak512_cpu_hash(int thr_id, int threads, uint32_t startNounce, uint64_t *d_hash, int order)
{
    const int threadsperblock = 128;

    dim3 grid(threads/threadsperblock);
    dim3 block(threadsperblock);

    size_t shared_size = 0;

    m7_keccak512_gpu_hash_120<<<grid, block, shared_size>>>(threads, startNounce, d_hash);
    MyStreamSynchronize(NULL, order, thr_id);
}

__host__ void m7_keccak512_setBlock_80(void *pdata)
{

	unsigned char PaddedMessage[128];
	uint8_t ending =0x01;
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage+80,ending,1); 
	memset(PaddedMessage+81, 0, 47); 
	cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 16*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	uint64_t* alt_data = (uint64_t*) pdata;
         uint64_t state[25];
		 for(int i=0;i<25;i++) {state[i]=0;}
		for (int i=0;i<9;i++) {state[i]  ^= alt_data[i];}
		keccak_block_host(state,cpu_RC);

		cudaMemcpyToSymbol(stateo, state, 25*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);

}

__host__ void ziftr_keccak512_setBlock_80(void *pdata)
{

	unsigned char PaddedMessage[128];
	uint8_t ending = 0x01;
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage + 80, ending, 1);
	memset(PaddedMessage + 81, 0, 47);
	cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	uint64_t* alt_data = (uint64_t*)pdata;
	uint64_t state[25];
	for (int i = 0; i<25; i++) { state[i] = 0; }
	state[0] = alt_data[0] & (~0xffff0000);
	for (int i = 1; i<9; i++) { state[i] = alt_data[i]; }
	keccak_block_host(state, cpu_RC);

	cudaMemcpyToSymbol(stateo, state, 25 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);

}


__host__ void ziftr_keccak512_cpu_hash_80(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash,uint32_t* d_test, int order)
{
    const int threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
    dim3 block(threadsperblock);

	const int threadsperblock2 = 128;
	dim3 grid2((threads + threadsperblock2 - 1) / threadsperblock2);
	dim3 block2(threadsperblock2);


    size_t shared_size = 0;
	if (compute_version[thr_id]>30) 
    ziftr_keccak512_gpu_hash_80<<<grid, block, shared_size>>>(threads, startNounce, d_hash, d_test);
    else 
	ziftr_keccak512_gpu_hash_80_v30 << <grid2, block2, shared_size >> >(threads, startNounce, d_hash, d_test);
    MyStreamSynchronize(NULL, order, thr_id);
}

__host__ void ziftr_keccak512_cpu_hash_80_round2(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, uint32_t* d_test, int order)
{
	const int threadsperblock = 256;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	const int threadsperblock2 = 128;
	dim3 grid2((threads + threadsperblock2 - 1) / threadsperblock2);
	dim3 block2(threadsperblock2);


	size_t shared_size = 0;
	if (compute_version[thr_id]>30)
	ziftr_keccak512_gpu_hash_80_round2 << <grid, block, shared_size >> >(threads, startNounce, d_hash, d_test);
    else
		ziftr_keccak512_gpu_hash_80_round2_v30 << <grid2, block2, shared_size >> >(threads, startNounce, d_hash, d_test);
 
	MyStreamSynchronize(NULL, order, thr_id);
}
