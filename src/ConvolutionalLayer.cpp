/*
 * ConvolutionalLayer.cpp
 *
 *  Created on: Nov 13, 2016
 *      Author: steve
 */

#include "ConvolutionalLayer.h"
#include <chrono>
#include <iostream>
#include <vector>

//#define PROFILE

ConvolutionalLayer::ConvolutionalLayer(int w, int h, int d, int stride,
                                       int padding, int K, int N)
    : width(w),
      height(h),
      dimension(d),
      stride(stride),
      padding(padding),
      K(K),
      N(N),
	  w(Eigen::MatrixXf::Random(N, K * K * dimension) * 0.1f),
	  b(Eigen::VectorXf::Zero(N)),
	  X(buildIM2COL()) {
}

// Calculate a sparse matrix that does the im2col transformation
ConvolutionalLayer::SpMat ConvolutionalLayer::buildIM2COL() const {
    int outWidth = getOutputWidth();
    int outHeight = getOutputHeight();

    SpMat x(outWidth * outHeight * K * K * dimension,
            width * height * dimension);
    std::vector<Triplet> tripletList;

    // iterate through the output pixels (each is one row in the im2col matrix)
    for (int yo = 0; yo < outHeight; ++yo) {
        for (int xo = 0; xo < outWidth; ++xo) {
            int outindex_base = (yo * outHeight + xo) * K * K * dimension;
            for (int yy = 0; yy < K; ++yy) {
                for (int xx = 0; xx < K; ++xx) {
                    for (int dd = 0; dd < dimension; ++dd) {
                        int xi = (xo - padding / 2) * stride + xx;
                        int yi = (yo - padding / 2) * stride + yy;
                        if (xi < 0 || yi < 0 || xi >= width || yi >= height)
                            continue;  // skip the invalid positions, in the
                                       // im2col matrix
                                       // they will be zero
                        int inindex = (yi * height + xi) * dimension + dd;
                        tripletList.push_back(Triplet(
                            outindex_base + (yy * K + xx) * dimension + dd,
                            inindex, 1));
                    }
                }
            }
        }
    }

    x.setFromTriplets(tripletList.begin(), tripletList.end());

    return x;
}

ConvolutionalLayer::~ConvolutionalLayer() {}

Eigen::MatrixXf ConvolutionalLayer::forward(const Eigen::MatrixXf& input) {
// inputs columns are the input images of the minibatch
// convert them to im2col with a matrix multiply

#ifdef PROFILE
    auto t0 = std::chrono::system_clock::now();
#endif

    Eigen::MatrixXf im2col = X * input;

#ifdef PROFILE
    auto t1 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = t1 - t0;
    std::cout << "Sparse mult" << elapsed_seconds.count() << std::endl;
#endif

    // reshape the im2col to K*K*D by W2*H2 matrix to prepare it for the
    // convolution
    inp = Eigen::MatrixXf(Eigen::Map<Eigen::MatrixXf>(
        im2col.data(), K * K * dimension,
        im2col.cols() * im2col.rows() / (K * K * dimension)));

#ifdef PROFILE
    auto t2 = std::chrono::system_clock::now();
    elapsed_seconds = t2 - t1;
    std::cout << "transpose" << elapsed_seconds.count() << std::endl;
#endif

    // perform the convolution with the im2col, which is this way a matrix
    // multiply
    // then add the bias.
    // the im2col is K*K*D by W2*H2 and our convolution filter bank is N by
    // K*K*D.
    Eigen::MatrixXf outputImages =
        w * inp + b * Eigen::VectorXf::Ones(inp.cols()).transpose();

#ifdef PROFILE
    auto t3 = std::chrono::system_clock::now();
    elapsed_seconds = t3 - t2;
    std::cout << "mult and add" << elapsed_seconds.count() << std::endl;
#endif

    // reshape our output images to the usual one-column-per-image format, just
    // like the input
    s = Eigen::MatrixXf(Eigen::Map<Eigen::MatrixXf>(
        outputImages.data(),
        outputImages.cols() * outputImages.rows() / input.cols(),
        input.cols()));

    // apply ReLU, just like in FC layer
    return s.cwiseMax(0);  // ReLU
}

Eigen::MatrixXf ConvolutionalLayer::backprop(const Eigen::MatrixXf& error) {
    // reshape error to the output image format (after the convolution)
    Eigen::MatrixXf sDiff = (s.array() > 0).select(error, 0);

    Eigen::MatrixXf sDiff_shape(Eigen::Map<Eigen::MatrixXf>(
        sDiff.data(), N, sDiff.rows() * sDiff.cols() / N));

    // calculate deltas for convolutional filter weights and biases
    deltaW = sDiff_shape * inp.transpose();
    deltaB = sDiff_shape * Eigen::VectorXf::Ones(sDiff_shape.cols());

    // calculate the backpropagated error, this is still in im2col format
    Eigen::MatrixXf backprop_im2col = w.transpose() * sDiff_shape;

    // reshape and convert back to input minibatch format
    Eigen::MatrixXf backprop_im2col_shape(Eigen::Map<Eigen::MatrixXf>(
        backprop_im2col.data(),
        backprop_im2col.cols() * backprop_im2col.rows() / error.cols(),
        error.cols()));

    Eigen::MatrixXf backprop_input = (X.transpose() * backprop_im2col_shape);

    return backprop_input;
}

void ConvolutionalLayer::applyWeightMod(float mu) {
    // TODO: ADAM
    w += deltaW * mu;
    b += deltaB * mu;
}

int ConvolutionalLayer::getOutputSize() const {
    return getOutputWidth() * getOutputHeight() * getOutputDimension();
}

int ConvolutionalLayer::getOutputWidth() const {
    return (width + padding - K) / stride + 1;
}
int ConvolutionalLayer::getOutputHeight() const {
    return (height + padding - K) / stride + 1;
}

int ConvolutionalLayer::getOutputDimension() const { return N; }

void ConvolutionalLayer::save(std::ostream& out) const {
	out << 'C';
	out.write(reinterpret_cast<const char*>(&width), sizeof(int));
	out.write(reinterpret_cast<const char*>(&height), sizeof(int));
	out.write(reinterpret_cast<const char*>(&dimension), sizeof(int));
	out.write(reinterpret_cast<const char*>(&stride), sizeof(int));
	out.write(reinterpret_cast<const char*>(&padding), sizeof(int));
	out.write(reinterpret_cast<const char*>(&K), sizeof(int));
	out.write(reinterpret_cast<const char*>(&N), sizeof(int));
	out.write(reinterpret_cast<const char*>(w.data()), w.rows()*w.cols()*sizeof(float));
	out.write(reinterpret_cast<const char*>(b.data()), b.rows()*b.cols()*sizeof(float));
}

ConvolutionalLayer* ConvolutionalLayer::load(std::istream& in) {
	int width;
	int height;
	int dimension;
	int stride;
	int padding;
	int K;
	int N;
	in.read(reinterpret_cast<char*>(&width), sizeof(int));
	in.read(reinterpret_cast<char*>(&height), sizeof(int));
	in.read(reinterpret_cast<char*>(&dimension), sizeof(int));
	in.read(reinterpret_cast<char*>(&stride), sizeof(int));
	in.read(reinterpret_cast<char*>(&padding), sizeof(int));
	in.read(reinterpret_cast<char*>(&K), sizeof(int));
	in.read(reinterpret_cast<char*>(&N), sizeof(int));

	ConvolutionalLayer* c = new ConvolutionalLayer(width, height, dimension, stride, padding, K, N);
	in.read(reinterpret_cast<char*>(c->w.data()), c->w.rows()*c->w.cols()*sizeof(float));
	in.read(reinterpret_cast<char*>(c->b.data()), c->b.rows()*c->b.cols()*sizeof(float));
    return c;
}
