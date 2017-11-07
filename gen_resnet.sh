python caffe-tensorflow/convert.py caffemodels/ResNet-101-deploy.prototxt --caffemodel caffemodels/ResNet-101-model.caffemodel --code-output-path=tfmodels/ResNet101.py --data-output-path=tfmodels/ResNet101.npy
python caffe-tensorflow/convert.py caffemodels/ResNet-152-deploy.prototxt --caffemodel caffemodels/ResNet-152-model.caffemodel --code-output-path=tfmodels/ResNet152.py --data-output-path=tfmodels/ResNet152.npy
python caffe-tensorflow/convert.py caffemodels/ResNet-50-deploy.prototxt --caffemodel caffemodels/ResNet-50-model.caffemodel --code-output-path=tfmodels/ResNet50.py --data-output-path=tfmodels/ResNet50.npy


