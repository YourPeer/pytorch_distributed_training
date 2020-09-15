# pytorch_distributed_training
在单个gpu的情况下，训练一个epoch的精度可以达到百分之50，但训练时间为249秒。  

![simple_gpu.png](./image/simple_gpu.png)  

在六个gpu的情况下，训练一个epoch的精度虽然只有百分之41，但训练时间仅为115秒。  

![multi_gpu.png](./image/multi_gpu.png)  
