#ifndef FLASH_H
#define FLASH_H
#include <cuda_fp16.h>

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
// 后续更新

#endif // KERNEL_H