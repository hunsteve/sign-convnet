/*
 * FullyConnectedLayer.h
 *
 *  Created on: Nov 11, 2016
 *      Author: steve
 */

#ifndef FULLYCONNECTEDLAYER_H_
#define FULLYCONNECTEDLAYER_H_

#include <istream>
#include <ostream>
#include "Layer.h"
#include "ADAM.h"

class FullyConnectedLayer : public Layer {
   private:
    bool isLinear;
    Eigen::MatrixXf w;
    Eigen::VectorXf b;
    ADAM adamW;
    ADAM adamB;

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
    int getOutputSize() const;

    void save(std::ostream& out) const;
    static FullyConnectedLayer* load(std::istream& in);

    int getParameterCount() const;
    void gradientCheck(int index, float epsilon, float* originalValue, float* originalDelta);
    void gradientCheckReset(int index, float originalValue);
};

#endif /* FULLYCONNECTEDLAYER_H_ */
