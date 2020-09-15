# pytorch_distributed_training

测试步骤：  

1、创建data文件夹  
2、将代码文件里的CIFAR10函数中的download参数改为True  

  trainset=CIFAR10(data_dir, train=True, transform=transform, target_transform=None, download=True)
  testset = CIFAR10(data_dir, train=False,download=True, transform=transform)

测试结果：  

1、在单个gpu的情况下，训练一个epoch的精度可以达到百分之50，但训练时间为249秒。  

![simple_gpu.png](./image/simple_gpu.png)  

2、在六个gpu的情况下，训练一个epoch的精度虽然只有百分之41，但训练时间仅为115秒。  

![multi_gpu.png](./image/multi_gpu.png)  
