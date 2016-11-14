/*
 * FullyConnectedLayer.h
 *
 *  Created on: Nov 11, 2016
 *      Author: steve
 */

#ifndef FULLYCONNECTEDLAYER_H_
#define FULLYCONNECTEDLAYER_H_

#include "Layer.h"
#include <istream>
#include <ostream>

class FullyConnectedLayer : public Layer {
private:
	bool isLinear;
	Eigen::MatrixXf w;
	Eigen::VectorXf b;

	Eigen::MatrixXf s;
	Eigen::MatrixXf inp;
	Eigen::MatrixXf deltaW;
	Eigen::VectorXf deltaB;

public:
	FullyConnectedLayer(int previousSize, int size, bool isLinear);
	virtual ~FullyConnectedLayer();
	Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
	Eigen::MatrixXf backprop(const Eigen::MatrixXf& error);
	void applyWeightMod(float mu);
	int getOutputSize() const ;

	void save(std::ostream& out) const ;
	static FullyConnectedLayer* load(std::istream& in);
};

#endif /* FULLYCONNECTEDLAYER_H_ */
