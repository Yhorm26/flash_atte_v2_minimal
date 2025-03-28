# flash_atte_v2_minimal
flash_attention v2的简易实现

目前还没有处理边界问题和一些功能，只计算了Softmax(Q*K.T)*V

在A100 上编译：

```nvcc *.cu -o flash -arch=sm_80 -std=c++11```

运行： 

```./flash```
            
在A100上运行的结果：

Max shared memory: 49152, requested shared memory: 28672

kernel time: 10475.528320 us

Verify the result of kernel function

===================start verfiy===================

==================finish,error:0==================

kernel time: 1864.371216 us

Verify the result of kernel function

===================start verfiy===================

==================finish,error:0==================

Max shared memory: 49152, kernel_3 requested shared memory: 36864

kernel time: 799.848877 us

Verify the result of kernel function

===================start verfiy===================

==================finish,error:0==================

kernel time: 338.784332 us

Verify the result of kernel function

===================start verfiy===================

==================finish,error:0==================




