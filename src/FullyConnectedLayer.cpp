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

int FullyConnectedLayer::getOutputSize() const {
	return w.rows();
}

void FullyConnectedLayer::save(std::ostream& out) const {
	out << 'F';
	int size = w.rows();
	int previousSize = w.cols();
	out.write(reinterpret_cast<const char*>(&isLinear), sizeof(bool));
	out.write(reinterpret_cast<const char*>(&size), sizeof(int));
	out.write(reinterpret_cast<const char*>(&previousSize), sizeof(int));
	out.write(reinterpret_cast<const char*>(w.data()), size*previousSize*sizeof(float));
	out.write(reinterpret_cast<const char*>(b.data()), size*sizeof(float));
}

FullyConnectedLayer* FullyConnectedLayer::load(std::istream& in) {
	int size;
	int previousSize;
	bool isLinear;
	in.read(reinterpret_cast<char*>(&isLinear), sizeof(bool));
	in.read(reinterpret_cast<char*>(&size), sizeof(int));
	in.read(reinterpret_cast<char*>(&previousSize), sizeof(int));
    FullyConnectedLayer* f = new FullyConnectedLayer(previousSize, size, isLinear);
    in.read(reinterpret_cast<char*>(f->w.data()), size*previousSize*sizeof(float));
	in.read(reinterpret_cast<char*>(f->b.data()), size*sizeof(float));

    return f;
}
