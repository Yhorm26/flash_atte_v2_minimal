#include <stdio.h>
#include "flash.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
using namespace nvcuda;

#define HMMA16816F32(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,  %1,  %2,  %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" : "+f"(RD0), "+f"(RD1), "+f"(RD2), "+f"(RD3) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "f"(RC0), "f"(RC1), "f"(RC2), "f"(RC3))
#define LDMATRIX_X2(R0, R1, addr) asm volatile( "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"  : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

__device__ inline uint32_t pack_float_to_uint32(float num1, float num2);

__global__
void forward_kernel_4(mykernelParamType2 param) {
    int tx = threadIdx.x;
    int warp_id = tx / 32; int lane_id = tx % 32;
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z; 

    int kv_offset = (bz * gridDim.y * param.N * param.d) + (by * param.N * param.d);
    int q_offset = kv_offset + bx * param.d * param.Br;

    half*  Q = param.Q + q_offset;
    half*  K = param.K + kv_offset;
    half*  V = param.V + kv_offset;
    float* O = param.O + q_offset;

    extern __shared__ half sram[];
    int tile_size = param.Bc * param.d;
    half* Qj     = sram;
    half* Kj     = &sram[param.Br * param.d];
    half* Vj     = &sram[param.Br * param.d + tile_size];

    float row_l_prev1 = 0;
    float row_l_prev2 = 0;
    float row_m_prev1 = -INFINITY;
    float row_m_prev2 = -INFINITY;

    const int load_Q_num  = param.Br * param.d / blockDim.x;
    const int load_KV_num = param.Bc * param.d / blockDim.x;
    
    uint32_t a_frag[4];
    uint32_t b_frag[4];
    float    c_frag[Bc4/16][8];
    uint32_t d_frag[Bc4/16][4];

    #pragma unroll
    for (int i = 0; i < load_Q_num; i++){
        Qj[tx * load_Q_num + i] = Q[tx * load_Q_num + i];
    }

    __syncthreads();

    for (int i = 0; i < param.Tc; i++){
        
        #pragma unroll
        for (int j = 0; j < load_KV_num; j++){
            Kj[tx * load_KV_num + j] = K[tx * load_KV_num + j];
            Vj[tx * load_KV_num + j] = V[tx * load_KV_num + j];
        }
        
        K += tile_size;
        V += tile_size;

        __syncthreads();
        
        // S = QK^T
        memset(c_frag, 0, sizeof(c_frag));
        #pragma unroll
        for (int x = 0; x < param.d / 16; x++){
            uint32_t aOffsetPtr = __cvta_generic_to_shared(&Qj[warp_id*param.d*16+x*16+(lane_id%16)*param.d+(lane_id/16)*8]);
            LDMATRIX_X4(a_frag[0], a_frag[1], a_frag[2], a_frag[3], aOffsetPtr);
            #pragma unroll
            for (int y = 0; y < param.Bc / 16; y++){
                uint32_t bOffsetPtr = __cvta_generic_to_shared(&Kj[y*param.d*16+x*16+(lane_id%16)*param.d+(lane_id/16)*8]);
                LDMATRIX_X4(b_frag[0], b_frag[2], b_frag[1], b_frag[3], bOffsetPtr);
                __syncwarp();

                HMMA16816F32(c_frag[y][0], c_frag[y][1], c_frag[y][4], c_frag[y][5], \
                             a_frag[0], a_frag[1], a_frag[2], a_frag[3], \
                             b_frag[0], b_frag[1], \
                             c_frag[y][0], c_frag[y][1], c_frag[y][4], c_frag[y][5]);

                HMMA16816F32(c_frag[y][2], c_frag[y][3], c_frag[y][6], c_frag[y][7], \
                             a_frag[0], a_frag[1], a_frag[2], a_frag[3], \
                             b_frag[2], b_frag[3], \
                             c_frag[y][2], c_frag[y][3], c_frag[y][6], c_frag[y][7]);
            }
        }

        __syncthreads();

        float row_m1 = -INFINITY;
        float row_m2 = -INFINITY;
        #pragma unroll
        for (int x = 0; x < param.Bc / 16; x++){
            #pragma unroll
            for (int y = 0; y < 4; y++){
                c_frag[x][y]     *= param.softmax_scale;
                c_frag[x][y + 4] *= param.softmax_scale;
                if (c_frag[x][y]     > row_m1)    row_m1 = c_frag[x][y]    ;
                if (c_frag[x][y + 4] > row_m2)    row_m2 = c_frag[x][y + 4];
            }
        }

        #pragma unroll
        for (int x = 3; x >= 1; x /= 2){
            float row_m_other = __shfl_xor_sync(0xffffffff, row_m1, x, 4);
            row_m1 = fmaxf(row_m1, row_m_other);
            row_m_other = __shfl_xor_sync(0xffffffff, row_m2, x, 4);
            row_m2 = fmaxf(row_m2, row_m_other);
        }

        float row_l1 = 0;
        float row_l2 = 0;
        #pragma unroll
        for (int x = 0; x < param.Bc / 16; x++) {
            #pragma unroll
            for (int y = 0; y < 4; y++){
                c_frag[x][y] = __expf(c_frag[x][y] - row_m1);
                row_l1 += c_frag[x][y];
                c_frag[x][y + 4] = __expf(c_frag[x][y + 4] - row_m2);
                row_l2 += c_frag[x][y + 4];
            }
        }

        #pragma unroll
        for (int x = 3; x >= 1; x /= 2){
            float row_l_other = __shfl_xor_sync(0xffffffff, row_l1, x, 4);
            row_l1 += row_l_other;
            row_l_other = __shfl_xor_sync(0xffffffff, row_l2, x, 4);
            row_l2 += row_l_other;
        }

        float row_m_new1 = fmaxf(row_m1, row_m_prev1);
        float row_m_new2 = fmaxf(row_m2, row_m_prev2);
        float row_l_new1 = (__expf(row_m_prev1 - row_m_new1) * row_l_prev1) + (__expf(row_m1 - row_m_new1) * row_l1);
        float row_l_new2 = (__expf(row_m_prev2 - row_m_new2) * row_l_prev2) + (__expf(row_m2 - row_m_new2) * row_l2);

        #pragma unroll
        for (int x = 0; x < param.Bc / 16; x++){
            #pragma unroll
            for (int y = 0; y < 4; y++){
                d_frag[x][y] = pack_float_to_uint32(c_frag[x][2*y], c_frag[x][2*y+1]);
            }
        }

        __syncthreads();

        float factor1 = 1 / row_l_new1;
        float factor2 = row_l_prev1 * __expf(row_m_prev1 - row_m_new1);
        float factor3 = __expf(row_m1 - row_m_new1);

        float factor4 = 1 / row_l_new2;
        float factor5 = row_l_prev2 * __expf(row_m_prev2 - row_m_new2);
        float factor6 = __expf(row_m2 - row_m_new2);

        // S = S * V
        #pragma unroll
        for (int x = 0; x < param.d / 16; x++){
            memset(c_frag, 0, sizeof(c_frag));
            #pragma unroll
            for(int y = 0; y < param.Bc / 16; y++){
                uint32_t bOffsetPtr = __cvta_generic_to_shared(&Vj[y*param.d*16+x*16+(lane_id%16)*param.d]);
                LDMATRIX_X2_T(b_frag[0], b_frag[1], bOffsetPtr);
                bOffsetPtr = __cvta_generic_to_shared(&Vj[y*param.d*16+x*16+(lane_id%16)*param.d+8]);;
                LDMATRIX_X2_T(b_frag[2], b_frag[3], bOffsetPtr);

                HMMA16816F32(c_frag[0][0], c_frag[0][1], c_frag[0][4], c_frag[0][5], \
                             d_frag[y][0], d_frag[y][2], d_frag[y][1], d_frag[y][3], \
                             b_frag[0], b_frag[1], \
                             c_frag[0][0], c_frag[0][1], c_frag[0][4], c_frag[0][5]);

                HMMA16816F32(c_frag[0][2], c_frag[0][3], c_frag[0][6], c_frag[0][7], \
                             d_frag[y][0], d_frag[y][2], d_frag[y][1], d_frag[y][3], \
                             b_frag[2], b_frag[3], \
                             c_frag[0][2], c_frag[0][3], c_frag[0][6], c_frag[0][7]);

                __syncthreads();
            }

            int offset = warp_id * param.d * 16 + x * 16 + (lane_id / 4) * param.d + (lane_id % 4) * 2;

            O[offset]     = factor1 * ((factor2 * O[offset    ]) + (factor3 * c_frag[0][0]));
            O[offset + 1] = factor1 * ((factor2 * O[offset + 1]) + (factor3 * c_frag[0][1]));
            O[offset + 8] = factor1 * ((factor2 * O[offset + 8]) + (factor3 * c_frag[0][2]));
            O[offset + 9] = factor1 * ((factor2 * O[offset + 9]) + (factor3 * c_frag[0][3]));

            offset += 8 * param.d;

            O[offset]     = factor4 * ((factor5 * O[offset    ]) + (factor6 * c_frag[0][4]));
            O[offset + 1] = factor4 * ((factor5 * O[offset + 1]) + (factor6 * c_frag[0][5]));
            O[offset + 8] = factor4 * ((factor5 * O[offset + 8]) + (factor6 * c_frag[0][6]));
            O[offset + 9] = factor4 * ((factor5 * O[offset + 9]) + (factor6 * c_frag[0][7]));
        }

        __syncthreads();

        row_l_prev1 = row_l_new1;
        row_l_prev2 = row_l_new2;
        row_m_prev1 = row_m_new1;
        row_m_prev2 = row_m_new2;
    }
}


__device__ inline uint32_t pack_float_to_uint32(float num1, float num2) {
    half a = __float2half(num1);
    half b = __float2half(num2);

    uint16_t a_bits = __half_as_ushort(a);
    uint16_t b_bits = __half_as_ushort(b);

    return (static_cast<uint32_t>(b_bits) << 16u) | a_bits;
}