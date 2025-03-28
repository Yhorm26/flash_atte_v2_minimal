#include <stdio.h>
#include "flash.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
using namespace nvcuda;

__device__ inline void load_smem_to_registers(uint32_t (&a_frag)[4], uint32_t (&b_frag)[4], const half* a_ptr, const half* b_ptr);
__device__ inline void mma(float (&c)[8], const uint32_t (&a)[4], const uint32_t (&b)[4]);
__device__ inline void store_mma_result(float (&c)[8], half* ptr, int M, int lane_id);
__device__ inline void atomicMaxFloat(float* addr, float value);


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
    half* Qj = sram;
    half* Kj = &sram[tile_size * 2];
    half* Vj = &sram[tile_size * 3];
    half* S  = &sram[tile_size * 4];

    __shared__ float l_prev[64];
    __shared__ float m_prev[64];
    __shared__ float l_new[64];
    __shared__ float m_new[64];

    if(tx < 64){
        l_prev[tx] = 0.0f;
        m_prev[tx] = -INFINITY;
    }
    
    uint32_t a_frag[4];
    uint32_t b_frag[4];
    float c_frag[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll
    for (int i = 0; i < param.d / 4; i++){
        Qj[tx * (param.d / 4) + i] = Q[tx * (param.d / 4) + i];
    }

    __syncthreads();

    for (int i = 0; i < param.Tc; i++){
        
        #pragma unroll
        for (int j = 0; j < param.d / 8; j++){
            Kj[(tx * param.d / 8) + j] = K[(tx * param.d / 8) + j];
        }

        #pragma unroll
        for (int j = 0; j < param.d / 8; j++){
            Vj[(tx / 8) + ((tx % 8) * 8 + j) * param.Bc] = V[(tx * param.d / 8) + j];
        }
        
        K += tile_size;
        V += tile_size;

        __syncthreads();
        
        // S = QK^T
        memset(c_frag, 0, sizeof(c_frag));
        #pragma unroll
        for (int x = 0; x < param.d / 16; x++){
            const half* aOffsetPtr = Qj + (warp_id / 2) * param.d * 16 + 16 * x + (tx % 16) * (param.d / 2) + tx / 16 * 4;
            const half* bOffsetPtr = Kj + (warp_id % 2) * param.d * 16 + 16 * x + (tx % 16) * (param.d / 2) + tx / 16 * 4;

            load_smem_to_registers(a_frag, b_frag, aOffsetPtr, bOffsetPtr);

            mma(c_frag, a_frag, b_frag);
        }

        int lm_offset = (warp_id / 2) * 16 + lane_id / 4;

        #pragma unroll
        for (int iter = 0; iter < 2; iter++){

            // row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int x = 0; x < 4; x++){
                c_frag[x + iter * 4] *= param.softmax_scale;
                if (c_frag[x + iter * 4] > row_m)    row_m = c_frag[x + iter * 4];
            }

            #pragma unroll
            for (int x = 3; x >= 1; x /= 2){
                float row_m_other = __shfl_xor_sync(0xffffffff, row_m, x, 4);
                row_m = fmaxf(row_m, row_m_other);
            }

            if(lane_id == 0){
                atomicMaxFloat(m_new + lm_offset + iter * 8, row_m);
            }
            
            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            #pragma unroll
            for (int x = 0; x < 4; x++) {
                c_frag[x + iter * 4] = __expf(c_frag[x + iter * 4] - row_m);
                row_l += c_frag[x + iter * 4];
            }

            #pragma unroll
            for (int x = 3; x >= 1; x /= 2){
                float row_l_other = __shfl_xor_sync(0xffffffff, row_l, x, 4);
                row_l += row_l_other;
            }

            if(lane_id == 0){
                atomicMaxFloat(l_new + lm_offset + iter * 8, row_l);
            }
        }

        half* cOffsetPtr = S + (warp_id / 2) * param.Bc * 16 + (warp_id % 2) * 16;
        store_mma_result(c_frag, cOffsetPtr, param.Bc, lane_id);

        __syncthreads();

        float row_m_new1 = fmaxf(m_new[lm_offset], m_prev[lm_offset]);
        float row_m_new2 = fmaxf(m_new[lm_offset + 8], m_prev[lm_offset + 8]);
        float row_l_new1 = fmaxf(l_new[lm_offset], l_prev[lm_offset]);
        float row_l_new2 = fmaxf(l_new[lm_offset + 8], l_prev[lm_offset + 8]);

        float factor1 = 1 / row_l_new1;
        float factor2 = l_prev[lm_offset] * __expf(m_prev[lm_offset] - row_m_new1);
        float factor3 = __expf(m_prev[lm_offset] - row_m_new1);

        float factor4 = 1 / row_l_new2;
        float factor5 = l_prev[lm_offset + 8] * __expf(m_prev[lm_offset + 8] - row_m_new2);
        float factor6 = __expf(m_prev[lm_offset + 8] - row_m_new2);

        // S = S * V
        #pragma unroll
        for (int x = 0; x < (param.d / 2) / 16; x++){
            memset(c_frag, 0, sizeof(c_frag));
            #pragma unroll
            for(int y = 0; y < param.Bc / 16; y++){
                const half* aOffsetPtr = S + (warp_id / 2) * param.Bc * 16 + 16 * y + (tx % 16) * (param.Bc / 2) + tx / 16 * 4;
                const half* bOffsetPtr = Vj + (warp_id % 2) * (param.d / 2) * param.Bc + x * param.Bc * 16 + y * 16 + + (tx % 16) * (param.Bc / 2) + tx / 16 * 4;

                load_smem_to_registers(a_frag, b_frag, aOffsetPtr, bOffsetPtr);

                mma(c_frag, a_frag, b_frag);
            }

            int offset = (warp_id / 2) * param.d * 16 + (warp_id % 2) * param.d / 2  \
                        + x * 16 + (lane_id / 4) * param.d + (lane_id % 4) * 2;

            O[offset]     = factor1 * ((factor2 * O[offset]) + (factor3 * c_frag[0]));
            O[offset + 1] = factor1 * ((factor2 * O[offset]) + (factor3 * c_frag[1]));
            O[offset + 2] = factor1 * ((factor2 * O[offset]) + (factor3 * c_frag[2]));
            O[offset + 3] = factor1 * ((factor2 * O[offset]) + (factor3 * c_frag[3]));

            offset += 8 * param.d;

            O[offset]     = factor4 * ((factor5 * O[offset]) + (factor6 * c_frag[4]));
            O[offset + 1] = factor4 * ((factor5 * O[offset]) + (factor6 * c_frag[5]));
            O[offset + 2] = factor4 * ((factor5 * O[offset]) + (factor6 * c_frag[6]));
            O[offset + 3] = factor4 * ((factor5 * O[offset]) + (factor6 * c_frag[7]));
        }

        __syncthreads();

        if (tx < 64){
            l_prev[tx] = l_new[tx];
            m_prev[tx] = m_new[tx];
        }
        __syncthreads();
    }
}


__device__ inline void load_smem_to_registers(
    uint32_t (&a_frag)[4],
    uint32_t (&b_frag)[4],
    const half* a_ptr,
    const half* b_ptr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "l"(__cvta_generic_to_shared(a_ptr))
    );
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
        : "l"(__cvta_generic_to_shared(b_ptr))
    );
}


__device__ inline void mma(
    float (&c)[8],
    const uint32_t (&a)[4],
    const uint32_t (&b)[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[4]), "+f"(c[5])
        : "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
          "r"(b[0]),  "r"(b[2])
    );

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%0, %1, %2, %3};\n"
        : "+f"(c[2]), "+f"(c[3]), "+f"(c[6]), "+f"(c[7])
        : "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
          "r"(b[1]),  "r"(b[3])
    );
}


__device__ inline void store_mma_result(
    float (&c)[8],
    half* ptr,
    int M,
    int lane_id)
{
    int offset = (lane_id / 4) * M + (lane_id % 4) * 2;
    ptr[offset]     = __float2half(c[0]);
    ptr[offset + 1] = __float2half(c[1]);
    ptr[offset + 8] = __float2half(c[2]);
    ptr[offset + 9] = __float2half(c[3]);

    offset += 8 * M;

    ptr[offset]     = __float2half(c[4]);
    ptr[offset + 1] = __float2half(c[5]);
    ptr[offset + 8] = __float2half(c[6]);
    ptr[offset + 9] = __float2half(c[7]);
}


__device__ inline void atomicMaxFloat(float* addr, float value) {
    float old = *addr;
    while (value > old) {
        float temp = atomicCAS(reinterpret_cast<unsigned int*>(addr),
                               __float_as_int(old),
                               __float_as_int(value));
        old = __int_as_float(temp);
    }
}