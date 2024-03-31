# ckpt-20230330-1845.pth

resnet18
* about 80 epochs with learning rate 0.1, 5 epochs with learning rate 0.01
* Train Loss: 0.036 | Acc: 98.838% (49419/50000)
* Test Loss: 0.191 | Acc: 94.560% (9456/10000)

# resnet-3layer253-50epochs-lr01.pth
model: commit 2576dc10fcbb6a1f4bb13ad43d83968f76f49125 ThreeLayerResNet
TrainConfig(
    epoch_count=50,
    learning_rate=0.1,
    batch_size_train=512,
)

Best accuracy:
Epoch: 31/50
100%|██████████| 98/98 [00:44<00:00,  2.22it/s]
Train Loss: 0.197 | Acc: 93.21% (46603/50000)
 Test Loss: 0.383 | Acc: 88.13% (8813/10000)
saved checkpoint to ./checkpoint/ckpt.pth
[0.0941882815044347] is learning rate

# resnet-3layer253-resume-10epochs-lr001

resume training resnet-3layer253-50epochs-lr01.pth

TrainConfig(
    epoch_count=10,
    learning_rate=0.01,
    batch_size_train=128,
)

Train Loss: 0.126 | Acc: 95.71% (47855/50000)
 Test Loss: 0.277 | Acc: 91.49% (9149/10000)

# resnet-3layer253-resume-5epochs-lr0001

resume training resnet-3layer253-resume-10epochs-lr001

TrainConfig(
epoch_count=5,
learning_Rate=0.001,
batch_size_train=128
)

# resnet-3layer253-resume3-5epochs-lr00001

resume from resnet-3layer253-resume-5epochs-lr0001

TrainConfig(
    epoch_count=5,
    learning_Rate=0.0001,
    batch_size_train=128
)
Train Loss: 0.055 | Acc: 98.41% (49204/50000)
 Test Loss: 0.243 | Acc: 92.30% (9230/10000)
