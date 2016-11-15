#sign - convnet
Convolutional Neural Network library and a model to classify traffic signs

This is an Eclipse CDT project.
Tested under Ubuntu 14.04

It uses the following libraries and API-s:
 - POSIX (for directory traversing)
 - [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (for linear algebra)
 - C++11/chrono (for performance measurement)
 
##How to build a ConvNet:

    NN nn(samplesX.rows());
    nn.addConvLayer(52,52,3,1,2,3,32);
    nn.addConvLayer(2,1,3,64);
    nn.addConvLayer(2,1,3,128);
    nn.addFCLayer(1000);
    nn.addFCLayer(1000);
    nn.addFCLayer(samplesY.rows(), true);
    
##How to train it:

    nn.train(samplesX, samplesY, 80, 0.001f, 0.95f, 250, saveNNCallback);
    
where samplesX and samplesY are matrices storing the input and target output data, columntwise (one sample per column), saveNNCallback is a callback function that is called at the end of each epoch.

##How to classify an image of a traffic sign:
 - checkout the project
 - build with or without Eclipse CDT 
 - make sure nn.dat is in the working directory of sign-convnet
 - run sign-convnet with a path to a 52x52 24-bit bitmap as command line argument

e.g.:

    sign-convnet train-52x52/7/7_1234.bmp

This will output one number, the class of the image, between 0 and 11 (original class - 1)

##nn.dat details:
- input: 52x52 3 channel bitmap
- convolution layer: 32 3x3 with zero padding -> 52x52x32
- convolution layer: 64 3x3 with zero padding and stride = 2 -> 26x26x64
- convolution layer: 128 3x3 with zero padding and stride = 2 -> 13x13x128
- fully connected layer: 1000 neuron
- fully connected layer: 1000 neuron
- fully connected linear layer: 12 neuron -> outputs

Training parameters: 
 - max epoch count = 75
 - learning rate = 0.001
 - training samples: 57000
 - validation samples: 3000
 - minibatch size: 250

Training took 8 hours on all 4 cores of an Intel i5-3570K.
Performance: 92.6% accuracy (measured on the validation set) 


