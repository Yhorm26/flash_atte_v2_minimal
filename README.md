# flash_atte_v2_minimal
flash_attention v2的简易实现

目前还没有处理边界问题和一些功能，只计算了Softmax(Q*K.T)*V

在v100 上编译：

```nvcc *.cu -o flash -arch=sm_70```

运行： 

```./flash```
            
在V100上运行的结果：

Max shared memory: 49152, requested shared memory: 28672

kernel time: 7732.272461 us

Verify the result of kernel function

===================start verfiy===================

==================finish,error:0==================

kernel time: 1697.723877 us

Verify the result of kernel function

===================start verfiy===================

==================finish,error:0==================

Max shared memory: 49152, kernel_3 requested shared memory: 36864

kernel time: 311.693451 us

Verify the result of kernel function

===================start verfiy===================

==================finish,error:0==================



