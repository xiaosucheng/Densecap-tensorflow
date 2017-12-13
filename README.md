# Densecap-tensorflow

**Warning**
This project reproduces the paper [DenseCap](https://arxiv.org/pdf/1511.07571.pdf) with Tensorflow. But currently, I reproduce it with low difficulty.

1. **Total separate trainning stages.** I first trained the RPN model, then I trained the RNN model using best proposals getting from RPN model and corresponding sentences.
2. **RPN model.** I used RoI pooling layer any way.

## Getting started

1. **Get my project.** `$ git clone` the repo.
2. **Configure dependency.** You need to configure py-caffe, tensorflow and visual_genome. 
3. **Get the VGG-16 caffemodel.** I use the 16-layer [VGG network](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) to extract features.

## Using the pretrained model to predict on one image

Due to the time, I only trained the net with 10 pictures. If you'd like to see the result, please use the pretrained model to predict on 'images/*.jpg'. You only need to run the 'demo.py'.

## Train your own model
1. **Get the data.** Download the train data folder from [visual_genome](http://visualgenome.org/).
2. **Train RPN model.** Run 'RPN.py'.
3. **Train RNN model.** Run 'RNN.py'.
