/*
 * NN.cpp
 *
 *  Created on: Nov 11, 2016
 *      Author: steve
 */

#include "NN.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include <iostream>
#include <math.h>

NN::NN(int inputSize) : inputSize(inputSize) {
}

NN::~NN() {
	for(unsigned int i = 0; i < layers.size(); ++i) {
		delete layers[i];
	}
}

Eigen::MatrixXf NN::forward(const Eigen::MatrixXf& input) {
	Eigen::MatrixXf layerOutput = input;
	for(unsigned int i = 0; i < layers.size(); ++i) {
		layerOutput = layers[i]->forward(layerOutput);
	}

	return layerOutput;
}

void NN::calcDeltas(const Eigen::MatrixXf& error) {
	Eigen::MatrixXf backPropError = error;
	for (unsigned int i=layers.size()-1; i<layers.size(); --i) {
		backPropError = layers[i]->backprop(backPropError);
	}
}

void NN::applyWeightMod(float mu) {
	for (unsigned int i = 0; i < layers.size(); ++i) {
		layers[i]->applyWeightMod(mu);
	}
}

void NN::train(const Eigen::MatrixXf& trainX, const Eigen::MatrixXf& trainY, int maxEpoch, float mu, float ratio, int minibatchSize, bool isDebug) {
	int trainBatchCount = (int) floor((trainX.rows() / minibatchSize) * ratio);

	for(int epoch = 0; epoch < maxEpoch; ++epoch) {
		float trainErrorSum = 0;
		for(int batch=0; batch<trainBatchCount; ++batch) {
			Eigen::MatrixXf inp = trainX.block(batch * minibatchSize,0,minibatchSize,trainX.cols());
			Eigen::MatrixXf targ = trainY.block(batch * minibatchSize,0,minibatchSize,trainY.cols());
			Eigen::MatrixXf out = forward(inp);
			Eigen::MatrixXf err = targ - out;
			std::cout << "epoch: " << epoch << "/" << maxEpoch << " batch: " << batch << "/" << trainBatchCount << std::endl;
			calcDeltas(err);
			applyWeightMod(2*mu);
			//sumerrtr += err.meanSquared();
		}
		float validationErrorSum = 0;

		int validationSize = trainX.rows() - trainBatchCount * minibatchSize;
		Eigen::MatrixXf inp = trainX.block(trainBatchCount * minibatchSize,0,validationSize,trainX.cols());
		Eigen::MatrixXf targ = trainY.block(trainBatchCount * minibatchSize,0,validationSize,trainY.cols());
		Eigen::MatrixXf out = forward(inp);
		Eigen::MatrixXf err = targ - out;
		//sumerrvalid += err.meanSquared();

		if (isDebug) {
			std::cout << "train error:" << (trainErrorSum / trainBatchCount);
		}

		float MSE = validationErrorSum / validationSize;
		std::cout << "MSE:" << MSE << std::endl;
	}
}


void NN::addConvLayer(int w, int h, int d, int stride, int padding, int K, int N) {
	int lastSize = 0;
	if (!layers.size())
		lastSize = inputSize;
	else
		lastSize = layers.back()->getOutputSize();
	Layer* layer = new ConvolutionalLayer(w, h, d, stride, padding, K, N);
	assert(w*h*d == lastSize);
	layers.push_back(layer);
}


void NN::addConvLayer(int stride, int padding, int K, int N) {
	assert(layers.size());
	ConvolutionalLayer* lastConvLayer = dynamic_cast<ConvolutionalLayer*>(layers.back());
	assert(lastConvLayer);

	Layer* layer = new ConvolutionalLayer(lastConvLayer->getOutputWidth(), lastConvLayer->getOutputHeight(), lastConvLayer->getOutputDimension(), stride, padding, K, N);
	layers.push_back(layer);
}


void NN::addFCLayer(int size, bool isLinear) {
	int lastSize = 0;
	if (!layers.size())
		lastSize = inputSize;
	else
		lastSize = layers.back()->getOutputSize();
	Layer* layer = new FullyConnectedLayer(lastSize, size, isLinear);
	layers.push_back(layer);
}

void NN::save(std::ofstream& out) {
	out.write((char*) (&inputSize), sizeof(int));
	for(unsigned int i = 0; i < layers.size(); ++i) {
		layers[i]->save(out);
	}
}

NN NN::load(std::ifstream& in) {
	int inputSize;
	in.read((char*) (&inputSize), sizeof(int));
	NN nn(inputSize);
	while (true) {
	    char c;
	    in.read(&c,sizeof(char));
	    if( in.eof() ) break;
	    Layer* nextLayer;
	    if (c=='F')
	    	nextLayer = FullyConnectedLayer::load(in);
	    else if (c=='C')
	    	nextLayer = ConvolutionalLayer::load(in);
	    nn.layers.push_back(nextLayer);
	}

	return nn;
}

