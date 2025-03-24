#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <sys/time.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "flash.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

std::default_random_engine generator(18);
std::uniform_real_distribution<float> distribution(0.0f, 10.0f);


void verfiy(
    float* O, 
    float* O_host,
    int batch_size,
    int n_head,
    int seq_len,
    int head_embd,
    float range_of_error)
{
    int error=0;
    printf("===================start verfiy===================\n");
    for(int i=0;i<batch_size*n_head*seq_len*head_embd;i++)
    {
        float device_out = O_host[i];
        if((fabs(O_host[i] - O[i]))/O_host[i] > range_of_error || std::isnan(device_out) || std::isinf(device_out))
        {
            printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, O_host[i], O[i]);
            error++;
            break;
        }        
    }
    printf("==================finish,error:%d==================\n",error);
}


void attention_forward_cpu(
    float* Q, 
    float* K, 
    float* V, 
    int batch_size, 
    int num_heads, 
    int seq_len, 
    int head_dim, 
    float sqrt_head_dim, 
    float* output)
{

    const int head_size = seq_len * head_dim;
    
    // 临时存储注意力分数
    float* scores = new float[seq_len * seq_len];

    for (int b = 0; b < batch_size; ++b) {
        
        for (int h = 0; h < num_heads; ++h) {
            // 获取当前head的指针偏移量
            const int base_offset = b * num_heads * head_size + h * head_size;
            const float* Q_ptr = Q + base_offset;
            const float* K_ptr = K + base_offset;
            const float* V_ptr = V + base_offset;
            float* out_ptr = output + base_offset;

            // 1. 手动实现QK^T矩阵乘法
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < head_dim; ++k) {
                        sum += Q_ptr[i * head_dim + k] * K_ptr[j * head_dim + k];
                    }
                    scores[i * seq_len + j] = sum * sqrt_head_dim;
                }
            }

            // 2. Softmax计算
            for (int i = 0; i < seq_len; ++i) {
                float max_val = -INFINITY;
                float* row = scores + i * seq_len;
                
                // 计算行最大值
                for (int j = 0; j < seq_len; ++j) {
                    if (row[j] > max_val) max_val = row[j];
                }

                // 计算指数和
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    row[j] = expf(row[j] - max_val);
                    sum += row[j];
                }

                // 归一化
                for (int j = 0; j < seq_len; ++j) {
                    row[j] /= sum;
                }
            }

            // 4. 手动实现注意力加权矩阵乘法
            for (int i = 0; i < seq_len; ++i) {
                for (int k = 0; k < head_dim; ++k) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        sum += scores[i * seq_len + j] * V_ptr[j * head_dim + k];
                    }
                    out_ptr[i * head_dim + k] = sum;
                }
            }
        }
    }

    delete[] scores;
}


int main(){
    int batch_size = 2;
    int n_head = 2;
    int seq_len = 1024;
    int head_embd = 64;

    float *Q      = (float*)malloc(batch_size*n_head*seq_len*head_embd*sizeof(float));
    float *K      = (float*)malloc(batch_size*n_head*seq_len*head_embd*sizeof(float));
    float *V      = (float*)malloc(batch_size*n_head*seq_len*head_embd*sizeof(float));
    float *O      = (float*)malloc(batch_size*n_head*seq_len*head_embd*sizeof(float));
    float *O_host = (float*)malloc(batch_size*n_head*seq_len*head_embd*sizeof(float));

    half *Q_half = (half*)malloc(batch_size*n_head*seq_len*head_embd*sizeof(half));
    half *K_half = (half*)malloc(batch_size*n_head*seq_len*head_embd*sizeof(half));
    half *V_half = (half*)malloc(batch_size*n_head*seq_len*head_embd*sizeof(half));

    float *Q_device,*K_device,*V_device, *O_device;
    cudaMalloc((void**)&Q_device, batch_size*n_head*seq_len*head_embd*sizeof(float));
    cudaMalloc((void**)&K_device, batch_size*n_head*seq_len*head_embd*sizeof(float));
    cudaMalloc((void**)&V_device, batch_size*n_head*seq_len*head_embd*sizeof(float));
    cudaMalloc((void**)&O_device, batch_size*n_head*seq_len*head_embd*sizeof(float));

    half *Q_device_half,*K_device_half,*V_device_half;
    cudaMalloc((void**)&Q_device_half, batch_size*n_head*seq_len*head_embd*sizeof(half));
    cudaMalloc((void**)&K_device_half, batch_size*n_head*seq_len*head_embd*sizeof(half));
    cudaMalloc((void**)&V_device_half, batch_size*n_head*seq_len*head_embd*sizeof(half));

    for(int i = 0; i < batch_size*n_head*seq_len*head_embd; i++)
    {
        Q[i] = distribution(generator);
        K[i] = distribution(generator);
        V[i] = distribution(generator);
        O[i] = 0.0f;

        Q_half[i] = __float2half(Q[i]);
        K_half[i] = __float2half(K[i]);
        V_half[i] = __float2half(V[i]);
    }
    
    cudaMemcpy(Q_device, Q, batch_size*n_head*seq_len*head_embd*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(K_device, K, batch_size*n_head*seq_len*head_embd*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(V_device, V, batch_size*n_head*seq_len*head_embd*sizeof(float),cudaMemcpyHostToDevice);

    cudaMemcpy(Q_device_half, Q_half, batch_size*n_head*seq_len*head_embd*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(K_device_half, K_half, batch_size*n_head*seq_len*head_embd*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(V_device_half, V_half, batch_size*n_head*seq_len*head_embd*sizeof(half),cudaMemcpyHostToDevice);

    mykernelParamType param;
    param.Q             = Q_device;
    param.K             = K_device;
    param.V             = V_device;
    param.O             = O_device;
    param.N             = seq_len;
    param.d             = head_embd;
    param.Bc            = 32;
    param.Br            = 32;
    param.Tc            = ceil((float)seq_len / param.Bc);
    param.Tr            = ceil((float)seq_len / param.Br);
    param.softmax_scale = 1.0 / sqrt(head_embd);

    // 计算每个线程块所需的SRAM大小
    const int sram_size = (3 * param.Bc * head_embd * sizeof(float)) + (param.Bc * param.Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    // ************************************kernel_1***************************************************
    dim3 grid_dim(param.Tr, n_head, batch_size);
    dim3 block_dim(param.Bc);

    // 预热
    forward_kernel_1<<<grid_dim, block_dim, sram_size>>>(param);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(O_host, O_device, batch_size*n_head*seq_len*head_embd*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    float time_elapsed=0.0;

    for (int i = 0; i < 100; i++){
        forward_kernel_1<<<grid_dim, block_dim, sram_size>>>(param);
    }
    
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);

    printf("kernel_1 time: %f us\n", time_elapsed*1000 / 100);

    attention_forward_cpu(Q, K, V, batch_size, n_head, seq_len, head_embd, param.softmax_scale, O);

    printf("Verify the result of kernel_1 function\n");
    verfiy(O, O_host, batch_size, n_head, seq_len, head_embd, 0.001);

    // ************************************kernel_2*************************************************

    dim3 grid_dim2(param.Tr, n_head, batch_size);
    dim3 block_dim2(param.Bc * 8);

    // 预热
    forward_kernel_2<<<grid_dim2, block_dim2, sram_size>>>(param);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(O_host, O_device, batch_size*n_head*seq_len*head_embd*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    time_elapsed  = 0.0;

    for (int i = 0; i < 100; i++){
       forward_kernel_2<<<grid_dim2, block_dim2, sram_size>>>(param);
    }
    
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);

    printf("kernel_2 time: %f us\n", time_elapsed*1000 / 100);
    printf("Verify the result of kernel_2 function\n");
    verfiy(O, O_host, batch_size, n_head, seq_len, head_embd, 0.001);
    // ************************************kernel_3*************************************************
    mykernelParamType2 param2;
    param2.Q            = Q_device_half;
    param2.K            = K_device_half;
    param2.V            = V_device_half;
    param2.O            = O_device;
    param2.N            = seq_len;
    param2.d            = head_embd;
    param2.Bc            = 32;
    param2.Br            = 64;
    param2.Tc            = ceil((float)seq_len / param2.Bc);
    param2.Tr            = ceil((float)seq_len / param2.Br);
    param2.softmax_scale = 1.0 / sqrt(head_embd);

    dim3 grid_dim3(param2.Tr / 2, n_head, batch_size);
    dim3 block_dim3(param2.Bc * 8);
    const int sram_size2 = (4 * param2.Bc * head_embd * sizeof(half)) + param2.Bc * param2.Br * sizeof(half);

    // 预热
    forward_kernel_3<<<grid_dim3, block_dim3, sram_size2>>>(param2);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(O_host, O_device, batch_size*n_head*seq_len*head_embd*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    time_elapsed=0.0;

    for (int i = 0; i < 100; i++){
        forward_kernel_3<<<grid_dim3, block_dim3, sram_size2>>>(param2);
    }
    
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);

    printf("kernel_3 time: %f us\n", time_elapsed*1000 / 100);
    printf("Verify the result of kernel_3 function\n");
    verfiy(O, O_host, batch_size, n_head, seq_len, head_embd, 0.04);

    // *********************************************************************************************

    cudaFree(Q_device);
    cudaFree(K_device);
    cudaFree(V_device);
    cudaFree(O_device);
    cudaFree(Q_device_half);
    cudaFree(K_device_half);
    cudaFree(V_device_half);
    
    free(Q);
    free(K);
    free(V);
    free(O);
    free(O_host);
    free(Q_half);
    free(K_half);
    free(V_half);
    
    return 0;
}
