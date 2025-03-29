#ifndef FLASH_H
#define FLASH_H
#include <cuda_fp16.h>
#include "utils.h"

// 核函数
__global__ void forward_kernel_1(mykernelParamType   param);   // 最简单的实现
__global__ void forward_kernel_2(mykernelParamType   param);   // 寄存器优化
__global__ void forward_kernel_3(mykernelParamType2  param);   // Tensor core优化
__global__ void forward_kernel_4(mykernelParamType2  param);   // MMA和ldmatrix指令
__global__ void forward_kernel_5(mykernelParamType2  param);   // cp.async指令/异步拷贝
// 后续更新

#endif  // FLASH_H
