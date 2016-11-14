/*
 * ConvolutionalLayer.h
 *
 *  Created on: Nov 13, 2016
 *      Author: steve
 */

#ifndef CONVOLUTIONALLAYER_H_
#define CONVOLUTIONALLAYER_H_
#include "Layer.h"
#include "Eigen/Sparse"
#include <fstream>

typedef Eigen::SparseMatrix<float> SpMat;
typedef Eigen::Triplet<float> Triplet;

class ConvolutionalLayer : public Layer {
private:
	int width;
	int height;
	int dimension;
	int stride;
	int padding;
	int K;
	int N;
	Eigen::MatrixXf w;
	Eigen::VectorXf b;

	Eigen::MatrixXf s;
	Eigen::MatrixXf inp;
	Eigen::MatrixXf deltaW;
	Eigen::VectorXf deltaB;

	SpMat X;

	SpMat buildIM2COL();

public:
	ConvolutionalLayer(int w, int h, int d, int stride, int padding, int K, int N);
	virtual ~ConvolutionalLayer();
	Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
	Eigen::MatrixXf backprop(const Eigen::MatrixXf& error);
	void applyWeightMod(float mu);
	int getOutputSize();
	int getOutputWidth();
	int getOutputHeight();
	int getOutputDimension();

	void save(std::ofstream& out);
	static ConvolutionalLayer* load(std::ifstream& in);
};

#endif /* CONVOLUTIONALLAYER_H_ */
