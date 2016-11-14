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
#include <fstream>
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

	void train(const Eigen::MatrixXf& trainX, const Eigen::MatrixXf& trainY, int maxEpoch, float mu, float ratio, int minibatchSize, bool isDebug);

	void addConvLayer(int w, int h, int d, int stride, int padding, int K, int N);
	void addConvLayer(int stride, int padding, int K, int N);
	void addFCLayer(int size, bool isLinear=false);

	void save(std::ofstream& out);
	static NN load(std::ifstream& in);
};

#endif /* NN_H_ */
