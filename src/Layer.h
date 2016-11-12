/*
 * Layer.h
 *
 *  Created on: Nov 11, 2016
 *      Author: steve
 */

#ifndef LAYER_H_
#define LAYER_H_
#include "Eigen/Dense"

class Layer {
public:
	virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input) = 0;
	virtual Eigen::MatrixXf backprop(const Eigen::MatrixXf& error) = 0;
	virtual void applyWeightMod(float mu) = 0;
	virtual int getOutputSize() = 0;
};

#endif /* LAYER_H_ */
