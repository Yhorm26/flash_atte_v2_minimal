#include <stdio.h>
#include "flash.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

__global__
void forward_kernel_3(mykernelParamType2 param) {
    int tx = threadIdx.x;
    int warp_id = tx / 32;
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z; 

    int kv_offset = (bz * gridDim.y * param.N * param.d) + (by * param.N * param.d);
    int q_offset = kv_offset + bx * param.d * param.Br;

    half*  Q = param.Q + q_offset;
    half*  K = param.K + kv_offset;
    half*  V = param.V + kv_offset;
    float* O = param.O + q_offset;

    float row_m_prev = -INFINITY;
    float row_l_prev = 0;

    extern __shared__ half sram[];
    int tile_size = param.Bc * param.d;
    half* Qj     = sram;
    half* Kj     = &sram[tile_size * 2];
    half* Vj     = &sram[tile_size * 3];
    half* S_half = &sram[tile_size * 4];

    __shared__ float S_mem[64 * 64];
    float* S = S_mem;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> d_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;;
    
    float scores[8];

    #pragma unroll
    for (int i = 0; i < param.d / 4; i++){
        Qj[tx * (param.d / 4) + i] = Q[tx * (param.d / 4) + i];
    }
    __syncthreads();

    for (int i = 0; i < param.Tc; i++){
        
        #pragma unroll
        for (int j = 0; j < param.d / 8; j++){
            Kj[(tx * param.d / 8) + j] = K[(tx * param.d / 8) + j];
            Vj[(tx * param.d / 8) + j] = V[(tx * param.d / 8) + j];
        }
        
        K += tile_size;
        V += tile_size;

        __syncthreads();
        
        // S = QK^T
        float row_m = -INFINITY;
        wmma::fill_fragment(c_frag, 0.0f);

        #pragma unroll
        for (int x = 0; x < param.d / 16; x++){
            const half* aOffsetPtr = Qj + (warp_id / 2) * param.d * 16 + 16 * x;
            const half* bOffsetPtr = Kj + (warp_id % 2) * param.d * 16 + 16 * x;

            load_matrix_sync(a_frag, aOffsetPtr, 64);
            load_matrix_sync(b_frag, bOffsetPtr, 64);

            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        float* cOffsetPtr = S + (warp_id / 2) * param.Bc * 16 + (warp_id % 2) * 16;

        store_matrix_sync(cOffsetPtr, c_frag, param.Bc, wmma::mem_row_major);

        __syncthreads();

        // row_m = rowmax(S)
        for (int x = 0; x < 8; x++){
            scores[x] = S[tx * 8 + x] * param.softmax_scale;
            if (scores[x] > row_m)    row_m = scores[x];
        }

        
        #pragma unroll
        for (int x = 3; x >= 1; x /= 2){
            float row_m_other = __shfl_xor_sync(0xffffffff, row_m, x, 4);
            row_m = fmaxf(row_m, row_m_other);
        }

        // P = exp(S - row_m), row_l = rowsum(P)
        float row_l = 0;
        #pragma unroll
        for (int x = 0; x < 8; x++) {
            scores[x] = __expf(scores[x] - row_m);
            row_l += scores[x];
        }

        #pragma unroll
        for (int x = 3; x >= 1; x /= 2){
            float row_l_other = __shfl_xor_sync(0xffffffff, row_l, x, 4);
            row_l += row_l_other;
        }

        #pragma unroll
        for (int x = 0; x < 8; x++){
            S_half[tx * 8 + x] = __float2half(scores[x]);
        }
        __syncthreads();    

        float row_m_new = row_m_prev > row_m ? row_m_prev : row_m;
        float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

        // S = S * V
        #pragma unroll
        for (int x = 0; x < (param.d / 2) / 16; x++){
            wmma::fill_fragment(c_frag, 0.0f);
            #pragma unroll
            for(int y = 0; y < param.Bc / 16; y++){
                const half* aOffsetPtr = S_half + (warp_id / 2) * param.Bc * 16 + 16 * y;
                const half* dOffsetPtr = Vj + (warp_id % 2) * param.d / 2 + 16 * x + y * param.d * 16;

                load_matrix_sync(a_frag, aOffsetPtr, 32);
                load_matrix_sync(d_frag, dOffsetPtr, param.d);
                
                mma_sync(c_frag, a_frag, d_frag, c_frag);
            }

            float* cOffsetPtr = S + (warp_id / 2) * param.d * 16 + (warp_id % 2) * param.d / 2 + x * 16;

            store_matrix_sync(cOffsetPtr, c_frag, param.d, wmma::mem_row_major);
        }
    
        __syncthreads();

        #pragma unroll
        for (int x = 0; x < param.d * param.Br / blockDim.x; x++){
            O[tx * param.d * param.Br / blockDim.x + x] = (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[tx * param.d * param.Br / blockDim.x + x]) \
                + (__expf(row_m - row_m_new) * S[tx * param.d * param.Br / blockDim.x + x]));
        }

        __syncthreads();

        row_l_prev = row_l_new;
        row_m_prev = row_m_new;
    }
}