/*
 * ConvolutionalLayer.cpp
 *
 *  Created on: Nov 13, 2016
 *      Author: steve
 */

#include "ConvolutionalLayer.h"
#include <vector>
#include <chrono>
#include <iostream>

ConvolutionalLayer::ConvolutionalLayer(int w, int h, int d, int stride, int padding, int K, int N)
: width(w), height(h), dimension(d), stride(stride), padding(padding), K(K), N(N){
	//this->w = Eigen::MatrixXf::Random(K*K*dimension, N) * 0.1f;
	this->w = Eigen::MatrixXf::Ones(K*K*dimension, N);
	this->b = Eigen::VectorXf::Zero(N);
	this->X = buildIM2COL();
}

//Calculate a sparse matrix that does the im2col transformation
SpMat ConvolutionalLayer::buildIM2COL() {

	int outWidth = getOutputWidth();
	int outHeight = getOutputHeight();

	SpMat x(width * height * dimension, outWidth * outHeight * K * K * dimension);
	std::vector<Triplet> tripletList;

	//iterate through the output pixels (each is one row in the im2col matrix)
	for(int yo = 0; yo<outHeight; ++yo) {
		for(int xo = 0; xo<outWidth; ++xo)
		{
			int outindex_base = (yo * outHeight + xo) * K * K * dimension;
			for(int yy = 0; yy < K; ++yy) {
				for(int xx = 0; xx < K; ++xx) {
					for (int dd = 0; dd<dimension; ++dd) {
						int xi = (xo - padding / 2) * stride + xx;
						int yi = (yo - padding / 2) * stride + yy;
						if (xi < 0 || yi < 0 || xi >= width || yi >= height)
							continue;//skip the invalid positions, in the im2col matrix they will be zero
						int inindex = (yi * height + xi) * dimension + dd;
						tripletList.push_back(Triplet(inindex, outindex_base + (yy*K + xx)*dimension + dd, 1));
					}
				}
			}
		}
	}

	x.setFromTriplets(tripletList.begin(), tripletList.end());

	return x;
}


ConvolutionalLayer::~ConvolutionalLayer() {
}

Eigen::MatrixXf ConvolutionalLayer::forward(const Eigen::MatrixXf& input) {



	//inputs rows are the input images of the minibatch
	//convert them to im2col with a matrix multiply

/*auto t0 = std::chrono::system_clock::now();*/

	Eigen::MatrixXf im2col = input * X;

/*auto t1 = std::chrono::system_clock::now();
std::chrono::duration<double> elapsed_seconds = t1-t0;
std::cout << "Sparse mult" << elapsed_seconds.count() << std::endl;*/

	//reshape the im2col to W2*H2 by K*K*D matrix to prepare it for the convolution
	//note: Eigen stores matrices in column-order, thats why we need the transposes
	//TODO: possible performance degradation due to transposes

	im2col.transposeInPlace();
	inp = Eigen::MatrixXf(Eigen::Map<Eigen::MatrixXf>(im2col.data(), K*K*dimension, im2col.cols() * im2col.rows() / (K*K*dimension)));
	inp.transposeInPlace();

/*auto t2 = std::chrono::system_clock::now();
elapsed_seconds = t2-t1;
std::cout << "transpose" << elapsed_seconds.count() << std::endl;*/

	//perform the convolution with the im2col, which is this way a matrix multiply
	//then add the bias.
	//the im2col is W2*H2 by K*K*D and our convolution filter bank is K*K*D by N.
	Eigen::MatrixXf outputImages = inp * w + Eigen::VectorXf::Ones(inp.rows()) * b.transpose();

/*auto t3 = std::chrono::system_clock::now();
elapsed_seconds = t3-t2;
std::cout << "mult and add" << elapsed_seconds.count() << std::endl;*/


	//reshape our output images to the usual one-row-per-image format, just like the input
	outputImages.transposeInPlace();
	s = Eigen::MatrixXf(Eigen::Map<Eigen::MatrixXf>(outputImages.data(), outputImages.cols() * outputImages.rows() / input.rows(), input.rows()));
	s.transposeInPlace();

	//apply ReLU, just like in FC layer
	return s.cwiseMax(0); //ReLU
}

Eigen::MatrixXf ConvolutionalLayer::backprop(const Eigen::MatrixXf& error) {

	//reshape error to the output image format (after the convolution)
	Eigen::MatrixXf sDiff = (s.array() > 0).select(error,0);
	sDiff.transposeInPlace();
	Eigen::MatrixXf sDiff_shape(Eigen::Map<Eigen::MatrixXf>(sDiff.data(), N, sDiff.rows() * sDiff.cols() / N));
	sDiff_shape.transposeInPlace();

	//calculate deltas for convolutional filter weights and biases
	deltaW = inp.transpose() * sDiff_shape;
	deltaB = Eigen::VectorXf::Ones(sDiff_shape.rows()).transpose() * sDiff_shape;

	//calculate the backpropagated error, this is still in im2col format
	Eigen::MatrixXf backprop_im2col = sDiff_shape * w.transpose();

	//reshape and convert back to input minibatch format
	backprop_im2col.transposeInPlace();
	Eigen::MatrixXf backprop_im2col_shape(Eigen::Map<Eigen::MatrixXf>(backprop_im2col.data(), backprop_im2col.cols() * backprop_im2col.rows() / error.rows(), error.rows()));

	Eigen::MatrixXf backprop_input = (X * backprop_im2col_shape);
	backprop_input.transposeInPlace();

	return backprop_input;
}

void ConvolutionalLayer::applyWeightMod(float mu) {
	w += deltaW * mu;
	b += deltaB * mu;
}

int ConvolutionalLayer::getOutputSize() {
	return getOutputWidth() * getOutputHeight() * getOutputDimension();
}

int ConvolutionalLayer::getOutputWidth() {
	return (width + padding - K) / stride + 1;
}
int ConvolutionalLayer::getOutputHeight() {
	return (height + padding - K) / stride + 1;
}

int ConvolutionalLayer::getOutputDimension() {
	return N;
}

void ConvolutionalLayer::save(std::ofstream& out) {
	out << 'C';
}


ConvolutionalLayer* ConvolutionalLayer::load(std::ifstream& in) {
	ConvolutionalLayer* c = new ConvolutionalLayer(1, 1, 1, 1, 1, 1, 1);
	return c;
}
