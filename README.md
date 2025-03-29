# flash_attn_v2_minimal
flash_attention v2的简易实现

目前只计算了Softmax(Q*K.T)*V

在A100 , src目录下编译：

```nvcc -I ../include *.cu -o flash -arch=sm_80 -std=c++11```

运行： 

```./flash```
            
在A100上运行的结果：

Max shared memory: 49152, requested shared memory: 28672

kernel time: 12181.115234 us

Verify the result of kernel function

===================start verfiy===================

==================finish,error:0==================

kernel time: 4033.533447 us

Verify the result of kernel function

===================start verfiy===================

==================finish,error:0==================

Max shared memory: 49152, kernel_3 requested shared memory: 36864

kernel time: 796.037415 us

Verify the result of kernel function

===================start verfiy===================

==================finish,error:0==================

kernel time: 338.171844 us

Verify the result of kernel function

===================start verfiy===================

==================finish,error:0==================

kernel time: 276.157104 us

Verify the result of kernel function

===================start verfiy===================

==================finish,error:0==================





