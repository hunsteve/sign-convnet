/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: Nov 11, 2016
 *      Author: steve
 */

#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(int previousSize, int size,
                                         bool isLinear)
	: isLinear(isLinear),
	  w(Eigen::MatrixXf::Random(size, previousSize) * 0.1f),
	  b(Eigen::VectorXf::Zero(size)) {
}

FullyConnectedLayer::~FullyConnectedLayer() {}

Eigen::MatrixXf FullyConnectedLayer::forward(const Eigen::MatrixXf& input) {
    inp = input;
    s = w * input + b * Eigen::VectorXf::Ones(input.cols()).transpose();
    if (isLinear)
        return s;
    else
        return s.cwiseMax(0);  // ReLU
}

Eigen::MatrixXf FullyConnectedLayer::backprop(const Eigen::MatrixXf& error) {
    Eigen::MatrixXf sDiff;
    if (isLinear)
        sDiff = error;
    else
        sDiff = (s.array() > 0).select(error, 0);  // ReLU derivative

    deltaW = sDiff * inp.transpose();
    deltaB = sDiff * Eigen::VectorXf::Ones(sDiff.cols());

    return w.transpose() * sDiff;
}

void FullyConnectedLayer::applyWeightMod(float mu) {
    // TODO: ADAM
    w += deltaW * mu;
    b += deltaB * mu;
}

int FullyConnectedLayer::getOutputSize() const { return w.rows(); }

void FullyConnectedLayer::save(std::ostream& out) const { out << 'F'; }

FullyConnectedLayer* FullyConnectedLayer::load(std::istream& in) {
    FullyConnectedLayer* f = new FullyConnectedLayer(1, 1, false);
    return f;
}
