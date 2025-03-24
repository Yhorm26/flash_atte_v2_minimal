# flash_atte_v2_minimal
flash_attention v2的简易实现

目前还没有处理边界问题和一些功能，只计算了Softmax(Q*K.T)*V

在v100 上做了编译：nvcc *.cu -o flash -arch=sm_70

            运行： ./flash
            
在V100上运行的结果：

Max shared memory: 49152, requested shared memory: 28672

kernel_1 time: 7465.029297 us

Verify the result of kernel_1 function

===================start verfiy===================

==================finish,error:0==================

kernel_2 time: 1697.499512 us

Verify the result of kernel_2 function

===================start verfiy===================

==================finish,error:0==================

kernel_3 time: 385.806091 us

Verify the result of kernel_3 function

===================start verfiy===================

==================finish,error:0==================


