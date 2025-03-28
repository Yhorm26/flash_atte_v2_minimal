#ifndef FLASH_H
#define FLASH_H
#include <cuda_fp16.h>

#define BATCH_SIZE 2   // 任意正整数
#define N_HEAD 2       // 任意正整数
#define SEQ_LEN 1024   // 需要为所有要运行的kernel的Br的最小公倍数的正整数倍
#define HEAD_EMBD 64   // 需要为所有要运行的kernel的Bc的最小公倍数的正整数倍

#define Br1 32   // 用于kernel1和kernel2
#define Bc1 32   // 用于kernel1和kernel2
#define Br2 32   // 用于后续kernel
#define Bc2 64   // 用于后续kernel

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
    half*  Q;                            
    half*  K;                        
    half*  V;                           
    float*   O;         
    int      N;
    int      d;
    int      Tc;
    int      Tr;
    int      Bc;
    int      Br;
    float    softmax_scale;
}mykernelParamType2;


// 核函数
__global__ void forward_kernel_1(mykernelParamType   param);   // 最简单的实现
__global__ void forward_kernel_2(mykernelParamType   param);   // 寄存器优化
__global__ void forward_kernel_3(mykernelParamType2  param);   // Tensor core优化
__global__ void forward_kernel_4(mykernelParamType2  param);
// 后续更新

#endif // KERNEL_H