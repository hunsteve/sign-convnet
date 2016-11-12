/*
 * NN.cpp
 *
 *  Created on: Nov 11, 2016
 *      Author: steve
 */

#include "NN.h"
#include "FullyConnectedLayer.h"
#include <iostream>

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

void NN::train(const Eigen::MatrixXf& trainX, const Eigen::MatrixXf& trainY, int maxEpoch, float mu, float ratio, bool isDebug) {

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



