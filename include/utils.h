#ifndef UTILS_H
#define UTILS_H
#include <cstdint>
#include <cuda_fp16.h>

#define BATCH_SIZE 2   // 任意正整数
#define N_HEAD 2       // 任意正整数
#define SEQ_LEN 1024   // 需要为所有要运行的kernel的Br的最小公倍数的正整数倍
#define HEAD_EMBD 64   // 需要为所有要运行的kernel的Bc的最小公倍数的正整数倍
#define Br1 32   // 用于kernel1和kernel2
#define Bc1 32   // 用于kernel1和kernel2
#define Br2 64   // 用于后续kernel
#define Bc2 16   // 用于后续kernel
#define Br4 64
#define Bc4 16
// MMA
#define HMMA16816F32(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,  %1,  %2,  %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" : "+f"(RD0), "+f"(RD1), "+f"(RD2), "+f"(RD3) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "f"(RC0), "f"(RC1), "f"(RC2), "f"(RC3))
// LDMATRIX
#define LDMATRIX_X2(R0, R1, addr) asm volatile( "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"  : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
// gmem -> smem
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"l"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"l"(dst), "l"(src), "n"(bytes))

typedef struct mykernelParamType
{
    float*  Q;
    float*  K;
    float*  V;
    float*  O;
    int     N;
    int     d;
    int     Tc;
    int     Tr;
    int     Bc;
    int     Br;
    float   softmax_scale;
}mykernelParamType;


typedef struct mykernelParamType2
{
    half*    Q; 
    half*    K;
    half*    V;
    float*   O;
    int      N;
    int      d;
    int      Tc;
    int      Tr;
    int      Bc;
    int      Br;
    float    softmax_scale;
}mykernelParamType2;


__device__ inline uint32_t pack_float_to_uint32(float num1, float num2) {
    half a = __float2half(num1);
    half b = __float2half(num2);

    uint16_t a_bits = __half_as_ushort(a);
    uint16_t b_bits = __half_as_ushort(b);

    return (static_cast<uint32_t>(b_bits) << 16u) | a_bits;
}

#endif // UTILS_H