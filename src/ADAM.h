/*
 * ADAM.h
 *
 *  Created on: Nov 16, 2016
 *      Author: steve
 */

#ifndef ADAM_H_
#define ADAM_H_
#include "Eigen/Dense"

class ADAM {
private:
	Eigen::MatrixXf m;
	Eigen::MatrixXf v;

	float beta1=0.9f;
	float beta2=0.999f;
	float epsilon=1e-8;

public:
	ADAM(int rows, int cols);
	virtual ~ADAM();

	Eigen::MatrixXf getWeightModification(Eigen::MatrixXf gradient);
};

#endif /* ADAM_H_ */
