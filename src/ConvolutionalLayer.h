/*
 * ConvolutionalLayer.h
 *
 *  Created on: Nov 13, 2016
 *      Author: steve
 */

#ifndef CONVOLUTIONALLAYER_H_
#define CONVOLUTIONALLAYER_H_
#include <istream>
#include <ostream>
#include "Eigen/Sparse"
#include "Layer.h"
#include "ADAM.h"

class ConvolutionalLayer : public Layer {
   private:
	typedef Eigen::SparseMatrix<float> SpMat;
	typedef Eigen::Triplet<float> Triplet;
    const int width;
    const int height;
    const int dimension;
    const int stride;
    const int padding;
    const int K;
    const int N;
    Eigen::MatrixXf w;
    Eigen::VectorXf b;
    ADAM adamW;
    ADAM adamB;

    Eigen::MatrixXf s;
    Eigen::MatrixXf inp;
    Eigen::MatrixXf deltaW;
    Eigen::VectorXf deltaB;

    SpMat X;

    SpMat buildIM2COL() const;

   public:
    ConvolutionalLayer(int w, int h, int d, int stride, int padding, int K,
                       int N);
    virtual ~ConvolutionalLayer();
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    Eigen::MatrixXf backprop(const Eigen::MatrixXf& error);
    void applyWeightMod(float mu);
    int getOutputSize() const;
    int getOutputWidth() const;
    int getOutputHeight() const;
    int getOutputDimension() const;

    void save(std::ostream& out) const;
    static ConvolutionalLayer* load(std::istream& in);


    int getParameterCount() const;
    void gradientCheck(int index, float epsilon, float* originalValue, float* originalDelta);
    void gradientCheckReset(int index, float originalValue);
};

#endif /* CONVOLUTIONALLAYER_H_ */
