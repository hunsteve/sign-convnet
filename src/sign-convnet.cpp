//============================================================================
// Name        : convnet.cpp
// Author      : Istvan Engedy
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "Eigen/Dense"

#include "FullyConnectedLayer.h"
#include "NN.h"
using namespace Eigen;
using namespace std;

int main() {
	MatrixXf input = MatrixXf::Random(5,3);

	NN nn(3);
	nn.addFCLayer(20);
	nn.addFCLayer(12,true);

	cout << nn.forward(input) << endl;

	return 0;
}
