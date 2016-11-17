/*
 * ADAM.cpp
 *
 *  Created on: Nov 16, 2016
 *      Author: steve
 */

#include "ADAM.h"

ADAM::ADAM(int rows, int cols):
	m(Eigen::MatrixXf::Zero(rows,cols)),
	v(Eigen::MatrixXf::Zero(rows,cols)){
}

ADAM::~ADAM() {
}

Eigen::MatrixXf ADAM::getWeightModification(Eigen::MatrixXf gradient) {
	m = beta1 * m + (1-beta1) * gradient;
	v = beta2 * v + (1-beta2) * (gradient.array() * gradient.array()).matrix();

	Eigen::MatrixXf m_hat = m / (1-beta1);
	Eigen::MatrixXf v_hat = v / (1-beta2);

	return (m_hat.array() / (v_hat.array().sqrt() + epsilon));
}
