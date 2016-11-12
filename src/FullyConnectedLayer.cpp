/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: Nov 11, 2016
 *      Author: steve
 */

#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(int previousSize, int size, bool isLinear)
{
	this->isLinear = isLinear;
	w = Eigen::MatrixXf::Random(previousSize, size) * 0.1f;
	b = Eigen::VectorXf::Zero(size);
}

FullyConnectedLayer::~FullyConnectedLayer() {
}

Eigen::MatrixXf FullyConnectedLayer::forward(const Eigen::MatrixXf& input){
	inp = input;
	s = input * w + Eigen::VectorXf::Ones(input.rows()) * b.transpose();
	if (isLinear)
		return s;
	else
		return s.cwiseMax(0); //ReLU
}

Eigen::MatrixXf FullyConnectedLayer::backprop(const Eigen::MatrixXf& error){
	Eigen::MatrixXf sDiff;
	if (isLinear)
		sDiff = error;
	else
		sDiff = (s.array() > 0).select(error,0); //ReLU derivative

	deltaW = inp.transpose() * sDiff;
	deltaB = sDiff;

	return sDiff * w.transpose();
}

void FullyConnectedLayer::applyWeightMod(float mu){
	w += deltaW * mu;
	b += deltaB * mu;
}

int FullyConnectedLayer::getOutputSize() {
	return w.cols();
}
