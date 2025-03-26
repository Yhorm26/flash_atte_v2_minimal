#include <stdio.h>
#include "flash.h"

__global__
void forward_kernel_2(mykernelParamType param) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;

    int kv_offset = (bz * gridDim.y * param.N * param.d) + (by * param.N * param.d);
    int q_offset = kv_offset + bx * param.d * param.Br;

    float* Q = param.Q + q_offset;
    float* K = param.K + kv_offset;
    float* V = param.V + kv_offset;
    float* O = param.O + q_offset;

    float Q_tile[8];
    float K_tile[8];
    float scores[Bc1 / 8];

    float row_m_prev = -INFINITY;
    float row_l_prev = 0;

    extern __shared__ float sram[];
    int tile_size = param.Bc * param.d;
    float* Qj = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    #pragma unroll
    for (int i = 0; i < param.d / 8; i++){
        Qj[tx * (param.d / 8) + i] = Q[tx * (param.d / 8) + i];
    }
    __syncthreads();

    unsigned int mask = 0xFF << (((tx % 32) / 8) * 8);

    for (int i = 0; i < param.Tc; i++){
        
        #pragma unroll
        for (int j = 0; j < param.d / 8; j++){
            Kj[(tx * param.d / 8) + j] = K[(tx * param.d / 8) + j];
            Vj[(tx * param.d / 8) + j] = V[(tx * param.d / 8) + j];
        }
        
        K += tile_size;
        V += tile_size;

        __syncthreads();

        // S = QK^T, row_m = rowmax(S)
        float row_m = -INFINITY;
        #pragma unroll
        for (int x = 0; x < Bc1 / 8; x++){
            scores[x] = 0.0f;
            #pragma unroll
            for (int y = 0; y < param.d / 8; y++){

                // 将Qj, Kj从SRAM加载到寄存器
                #pragma unroll
                for (int m = 0; m < 8; m++){
                    Q_tile[m] = Qj[(tx / 8) * param.d + y * 8 + m];
                     K_tile[m] = Kj[((tx % 8) * (Bc1 / 8) + x) * param.d + y * 8 + m];
                }
                #pragma unroll
                for (int m = 0; m < 8; m++){
                    scores[x] += Q_tile[m] * K_tile[m];
                }
            }
            scores[x] *= param.softmax_scale;
            if (scores[x] > row_m)    row_m = scores[x];
        }

        // row_m = rowmax(S)
        #pragma unroll
        for (int x = 7; x >= 1; x /= 2){
            float row_m_other = __shfl_xor_sync(mask, row_m, x, 32);
            row_m = fmaxf(row_m, row_m_other);
        }

        // P = exp(S - row_m), row_l = rowsum(P)
        float row_l = 0;
        #pragma unroll
        for (int x = 0; x < Bc1 / 8; x++) {
            scores[x] = __expf(scores[x] - row_m);
            row_l += scores[x];
        }

        #pragma unroll
        for (int x = 7; x >= 1; x /= 2){
            float row_l_other = __shfl_xor_sync(mask, row_l, x, 32);
            row_l += row_l_other;
        }

        #pragma unroll
        for (int x = 0; x < Bc1 / 8; x++){
            S[tx * Bc1 / 8 + x] = scores[x];
        }
        __syncthreads();

        float row_m_new = row_m_prev > row_m ? row_m_prev : row_m;
        float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

        float(&S_tile)[8] = Q_tile;
        float(&V_tile)[8] = K_tile;
        float result[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int x = 0; x < param.d / 8; x++) {
            #pragma unroll
            for (int y = 0; y < param.Bc / 8; y++) {
                #pragma unroll
                for (int m = 0; m < 8; m++){
                    S_tile[m] = S[(tx / 8) * param.Bc + y * 8 + m];
                    V_tile[m] = Vj[((tx % 8) * 8 + x) + (y * 8 + m) * param.d];
                }
                #pragma unroll
                for (int m = 0; m < 8; m++){
                    result[x] += S_tile[m] * V_tile[m];
                }
            }
            O[(tx * param.d / 8) + x] = (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[(tx * param.d / 8) + x]) \
                + (__expf(row_m - row_m_new) * result[x]));
        }
        __syncthreads();

        row_l_prev = row_l_new;
        row_m_prev = row_m_new;
    }
}