/*
 * NN.h
 *
 *  Created on: Nov 11, 2016
 *      Author: steve
 */

#ifndef NN_H_
#define NN_H_
#include <vector>
#include <string>
#include "Layer.h"

class NN {
private:
	std::vector<Layer*> layers;
	int inputSize;
public:
	NN(int inputSize);
	virtual ~NN();

	Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
	void calcDeltas(const Eigen::MatrixXf& error);
	void applyWeightMod(float mu);

	void train(const Eigen::MatrixXf& trainX, const Eigen::MatrixXf& trainY, int maxEpoch, float mu, float ratio, bool isDebug);

	void addConvLayer();
	void addFCLayer(int size, bool isLinear=false);
};

#endif /* NN_H_ */
