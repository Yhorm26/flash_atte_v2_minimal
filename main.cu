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
    float range_of_error)
{
    int error=0;
    printf("===================start verfiy===================\n");
    for(int i=0;i<BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD;i++)
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
    float sqrt_head_dim, 
    float* output)
{

    const int head_size = SEQ_LEN * HEAD_EMBD;
    
    // 临时存储注意力分数
    float* scores = new float[SEQ_LEN * SEQ_LEN];

    for (int b = 0; b < BATCH_SIZE; ++b) {
        
        for (int h = 0; h < N_HEAD; ++h) {
            // 获取当前head的指针偏移量
            const int base_offset = b * N_HEAD * head_size + h * head_size;
            const float* Q_ptr = Q + base_offset;
            const float* K_ptr = K + base_offset;
            const float* V_ptr = V + base_offset;
            float* out_ptr = output + base_offset;

            // 1. 手动实现QK^T矩阵乘法
            for (int i = 0; i < SEQ_LEN; ++i) {
                for (int j = 0; j < SEQ_LEN; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < HEAD_EMBD; ++k) {
                        sum += Q_ptr[i * HEAD_EMBD + k] * K_ptr[j * HEAD_EMBD + k];
                    }
                    scores[i * SEQ_LEN + j] = sum * sqrt_head_dim;
                }
            }

            // 2. Softmax计算
            for (int i = 0; i < SEQ_LEN; ++i) {
                float max_val = -INFINITY;
                float* row = scores + i * SEQ_LEN;
                
                // 计算行最大值
                for (int j = 0; j < SEQ_LEN; ++j) {
                    if (row[j] > max_val) max_val = row[j];
                }

                // 计算指数和
                float sum = 0.0f;
                for (int j = 0; j < SEQ_LEN; ++j) {
                    row[j] = expf(row[j] - max_val);
                    sum += row[j];
                }

                // 归一化
                for (int j = 0; j < SEQ_LEN; ++j) {
                    row[j] /= sum;
                }
            }

            // 4. 手动实现注意力加权矩阵乘法
            for (int i = 0; i < SEQ_LEN; ++i) {
                for (int k = 0; k < HEAD_EMBD; ++k) {
                    float sum = 0.0f;
                    for (int j = 0; j < SEQ_LEN; ++j) {
                        sum += scores[i * SEQ_LEN + j] * V_ptr[j * HEAD_EMBD + k];
                    }
                    out_ptr[i * HEAD_EMBD + k] = sum;
                }
            }
        }
    }

    delete[] scores;
}


void launchKernel(
    mykernelParamType param, 
    void (*kernel)(mykernelParamType), 
    int grid_x, int grid_y, int grid_z, 
    int block_x, 
    int sram_size, 
    float* O,
    float* O_host,
    float* O_device,
    float range_of_error) 
{
    dim3 grid_dim(grid_x, grid_y, grid_z);
    dim3 block_dim(block_x);

    // 预热
    kernel<<<grid_dim, block_dim, sram_size>>>(param);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(O_host, O_device, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    float time_elapsed=0.0;

    for (int i = 0; i < 100; i++){
        kernel<<<grid_dim, block_dim, sram_size>>>(param);
    }
    
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);

    printf("kernel time: %f us\n", time_elapsed*1000 / 100);
    printf("Verify the result of kernel function\n");

    verfiy(O, O_host, range_of_error);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void launchKernel(
    mykernelParamType2 param, 
    void (*kernel)(mykernelParamType2), 
    int grid_x, int grid_y, int grid_z, 
    int block_x, 
    int sram_size, 
    float* O,
    float* O_host,
    float* O_device,
    float range_of_error) 
{
    dim3 grid_dim(grid_x, grid_y, grid_z);
    dim3 block_dim(block_x);

    // 预热
    kernel<<<grid_dim, block_dim, sram_size>>>(param);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(O_host, O_device, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    float time_elapsed=0.0;

    for (int i = 0; i < 100; i++){
        kernel<<<grid_dim, block_dim, sram_size>>>(param);
    }
    
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);

    printf("kernel time: %f us\n", time_elapsed*1000 / 100);
    printf("Verify the result of kernel function\n");

    verfiy(O, O_host, range_of_error);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main(){
    float *Q      = (float*)malloc(BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(float));
    float *K      = (float*)malloc(BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(float));
    float *V      = (float*)malloc(BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(float));
    float *O      = (float*)malloc(BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(float));
    float *O_host = (float*)malloc(BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(float));

    half *Q_half = (half*)malloc(BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(half));
    half *K_half = (half*)malloc(BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(half));
    half *V_half = (half*)malloc(BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(half));

    float *Q_device,*K_device,*V_device, *O_device;
    cudaMalloc((void**)&Q_device, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(float));
    cudaMalloc((void**)&K_device, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(float));
    cudaMalloc((void**)&V_device, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(float));
    cudaMalloc((void**)&O_device, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(float));

    half *Q_device_half,*K_device_half,*V_device_half;
    cudaMalloc((void**)&Q_device_half, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(half));
    cudaMalloc((void**)&K_device_half, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(half));
    cudaMalloc((void**)&V_device_half, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(half));

    for(int i = 0; i < BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD; i++)
    {
        Q[i] = distribution(generator);
        K[i] = distribution(generator);
        V[i] = distribution(generator);
        O[i] = 0.0f;

        Q_half[i] = __float2half(Q[i]);
        K_half[i] = __float2half(K[i]);
        V_half[i] = __float2half(V[i]);
    }
    
    cudaMemcpy(Q_device, Q, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(K_device, K, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(V_device, V, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(float),cudaMemcpyHostToDevice);

    cudaMemcpy(Q_device_half, Q_half, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(K_device_half, K_half, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(V_device_half, V_half, BATCH_SIZE*N_HEAD*SEQ_LEN*HEAD_EMBD*sizeof(half),cudaMemcpyHostToDevice);

    mykernelParamType param;
    param.Q             = Q_device;
    param.K             = K_device;
    param.V             = V_device;
    param.O             = O_device;
    param.N             = SEQ_LEN;
    param.d             = HEAD_EMBD;
    param.Bc            = Bc1;
    param.Br            = Br1;
    param.Tc            = ceil(SEQ_LEN / param.Bc);
    param.Tr            = ceil(SEQ_LEN / param.Br);
    param.softmax_scale = 1.0 / sqrt(HEAD_EMBD);

    // 计算每个线程块所需的SRAM大小
    int sram_size = ((2 * param.Bc + param.Br) * HEAD_EMBD * sizeof(float)) + (param.Bc * param.Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);
    attention_forward_cpu(Q, K, V, param.softmax_scale, O);

    // ************************************kernel_1***************************************************************************
    launchKernel(param,forward_kernel_1, param.Tr, N_HEAD, BATCH_SIZE, param.Br, sram_size, O, O_host, O_device, 0.0001);
    // ************************************kernel_2***************************************************************************
    launchKernel(param,forward_kernel_2, param.Tr, N_HEAD, BATCH_SIZE, param.Br * 8, sram_size, O, O_host, O_device, 0.0001);
    // ************************************kernel_3***************************************************************************
    mykernelParamType2 param2;
    param2.Q             = Q_device_half;
    param2.K             = K_device_half;
    param2.V             = V_device_half;
    param2.O             = O_device;
    param2.N             = SEQ_LEN;
    param2.d             = HEAD_EMBD;
    param2.Bc            = Bc2;
    param2.Br            = Br2;
    param2.Tc            = ceil(SEQ_LEN / param2.Bc);
    param2.Tr            = ceil(SEQ_LEN / param2.Br);
    param2.softmax_scale = 1.0 / sqrt(HEAD_EMBD);

    int sram_size2 = ((2 * param2.Bc + param2.Br) * HEAD_EMBD * sizeof(half)) + param2.Bc * param2.Br * sizeof(half);
    printf("Max shared memory: %d, kernel_3 requested shared memory: %d \n", max_sram_size, sram_size + param.d * param.Br * 4);

    launchKernel(param2, forward_kernel_3, param2.Tr, N_HEAD, BATCH_SIZE, (param2.Br/16)*(param2.Bc/16)*32, sram_size2, O, O_host, O_device, 0.04);
    // ****************************************************************************************************************************

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