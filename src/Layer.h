/*
 * Layer.h
 *
 *  Created on: Nov 11, 2016
 *      Author: steve
 */

#ifndef LAYER_H_
#define LAYER_H_
#include <istream>
#include <ostream>
#include "Eigen/Dense"

class Layer {
   public:
	virtual ~Layer() {}
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input) = 0;
    virtual Eigen::MatrixXf backprop(const Eigen::MatrixXf& error) = 0;
    virtual void applyWeightMod(float mu) = 0;
    virtual int getOutputSize() const = 0;

    virtual void save(std::ostream& out) const = 0;
    virtual int getParameterCount() const = 0;
    virtual void gradientCheck(int index, float epsilon, float* originalValue, float* originalDelta) = 0;
    virtual void gradientCheckReset(int index, float originalValue) = 0;
};

#endif /* LAYER_H_ */
