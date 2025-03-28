#include <stdio.h>
#include "flash.h"

__global__
void forward_kernel_1(mykernelParamType param) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;

    // Q,K,V,O的偏移
    int kv_offset = (bz * gridDim.y * param.N * param.d) + (by * param.N * param.d);
    int q_offset = kv_offset + bx * param.d * param.Br;

    float* Q = param.Q + q_offset;
    float* K = param.K + kv_offset;
    float* V = param.V + kv_offset;
    float* O = param.O + q_offset;

    float row_m_prev = -INFINITY;
    float row_l_prev = 0;

    // 申请Q,K,V,S的SRAM
    extern __shared__ float sram[];
    int tile_size = param.Bc * param.d;  // size of Qj, Kj, Vj
    float* Qj = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    // 把Qj从全局内存加载到SRAM
    for (int i = 0; i < param.d; i++){
        Qj[tx * param.d + i] = Q[tx * param.d + i];
    }
    __syncthreads();

    for (int i = 0; i < param.Tc; i++){

        // 把Kj, Vj从全局内存加载到SRAM
        for (int j = 0; j < param.d; j++){
            Kj[(tx * param.d) + j] = K[(tx * param.d) + j];
            Vj[(tx * param.d) + j] = V[(tx * param.d) + j];
        }
        
        K += tile_size;
        V += tile_size;

        __syncthreads();

        // S = QK^T, row_m = rowmax(S)
        float row_m = -INFINITY;
        for (int y = 0; y < param.Bc; y++){
            float sum = 0;
            for (int x = 0; x < param.d; x++){
                sum += Qj[(tx * param.d) + x] * Kj[(y * param.d) + x];
            }
            sum *= param.softmax_scale;
            S[(param.Bc * tx) + y] = sum;

            if (sum > row_m)
                row_m = sum;
        }

        // P = exp(S - row_m), row_l = rowsum(P)
        float row_l = 0;
        for (int y = 0; y < param.Bc; y++) {
            S[(param.Bc * tx) + y] = __expf(S[(param.Bc * tx) + y] - row_m);
            row_l += S[(param.Bc * tx) + y];
        }

        // 更新m a和 l
        float row_m_new = row_m_prev > row_m ? row_m_prev : row_m;
        float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

        // 计算并写回O
        for (int x = 0; x < param.d; x++) {
            float pv = 0;  // Pij * Vj
            for (int y = 0; y < param.Bc; y++) {
                pv += S[(param.Bc * tx) + y] * Vj[(y * param.d) + x];
            }
            O[(tx * param.d) + x] = (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[(tx * param.d) + x]) \
                + (__expf(row_m - row_m_new) * pv));
        }
        

        __syncthreads();

        row_l_prev = row_l_new;
        row_m_prev = row_m_new;
    }
}