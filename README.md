caffe-tensorflow和SPCap/kaffe目录下的内容来自[ethreon](https://github.com/ethereon/caffe-tensorflow)，遵守原作者声明的协议。

### 用法

1. ResNet的.prototxt和.caffemodel文件从He的[onedrive](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)下载，也可以从我的[网盘](https://pan.baidu.com/s/1hsrfKtI)下载，将所有文件放在caffemodels目录下；
2. 修改几个prototxt文件第一行name，将“-”改为“_”，避免生成的python文件出错； 
3. 赋予执行权限，chmod +x gen_resnet.sh；
4. 执行./gen_resnet.sh将caffe模型转为tf模型。
5. 进入SPCup目录，执行test_converted_nets.py，输出[235]，即预测结果为第235类（从0开始记）在imagenet_classes.txt里搜索235，其对应的名字是German_shepherd，德国牧羊犬，预测正确。