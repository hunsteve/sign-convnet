/*
 * NN.cpp
 *
 *  Created on: Nov 11, 2016
 *      Author: steve
 */

#include "NN.h"
#include "FullyConnectedLayer.h"
#include <iostream>
#include <math.h>

NN::NN(int inputSize) : inputSize(inputSize) {
}

NN::~NN() {
	// TODO Auto-generated destructor stub
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
	for (unsigned int i=layers.size()-1; i>=0; --i) {
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
		std::cout << "MSE:" << MSE;
	}
}


void NN::addConvLayer() {
	//TODO: add convolutional layer
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



