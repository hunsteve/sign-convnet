//============================================================================
// Name        : convnet.cpp
// Author      : Istvan Engedy
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <iomanip>
#include "Eigen/Dense"

#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "NN.h"
#include "bitmap.h"
#include <ftw.h>
#include <algorithm>

using namespace Eigen;
using namespace std;

vector<string> files;

int collect_files(const char *filepath, const struct stat *info,
                  const int typeflag, struct FTW *pathinfo)
{
	if (typeflag == FTW_F) {
		string fn(filepath);
		if (!fn.compare(fn.length() - 4, 4, ".bmp"))
			files.push_back(fn);
	}
    return 0;
}

int filenameToClassIndex(string fn) {
    unsigned int from = fn.rfind("/") + 1;
    if (from == string::npos)
    	from = 0;
    unsigned int till = fn.rfind("_");
    int sampleClass;
    stringstream ss(fn.substr(from, till-from));
    ss >> sampleClass;
    sampleClass -= 1;

    return sampleClass;
}

void BMPFilesToSamples(vector<string> files, int features, int classes, MatrixXf* samplesX, MatrixXf* samplesY) {
	random_shuffle(files.begin(), files.end());


	*samplesX = MatrixXf(features, files.size());
	*samplesY = MatrixXf(classes, files.size());

	int i=0;
	for (vector<string>::iterator it=files.begin(); it!=files.end(); ++it) {

	    vector<unsigned char> data = readBMP(it->c_str());
	    typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix8u;

	    MatrixXf datacol = Map<Matrix8u>(data.data(), data.size(), 1).cast<MatrixXf::Scalar>();

	    datacol = (datacol - MatrixXf::Ones(data.size(), 1) * 128) / 128.0f; //normalize to near N(0,1)

	    samplesX->block(0,i,features,1) = datacol;
	    samplesY->block(0,i,classes,1) = MatrixXf::Zero(classes,1);
	    (*samplesY)(filenameToClassIndex(*it),i) = 1;
	    i++;
	    if (i%1000 == 0)
	    	cout << "Read " << i << " files." << endl;

/*
	    for(int x=0; x<52; ++x){
	    	for(int y=0; y<52; ++y){
	    		cout << HEX(data[(x + y*52)*3]+2) << " ";
			}
	    	cout << endl;
	    }*/
	}
}

int main() {
	Eigen::initParallel();

	cout << "Threads used: " << Eigen::nbThreads() << endl;
	if (Eigen::nbThreads() == 1)
		cout << "Warning! using only one thread! make sure to use -fopenmp with G++" << endl;

	/*Eigen::MatrixXf m(6,3);
	m << 1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19;

	cout << m << endl;

	m.transposeInPlace();
	MatrixXf v(Map<MatrixXf>(m.data(), m.cols()*m.rows() / 2, 2));
	v.transposeInPlace();

	cout << v << endl;
	 */


	const char* path = "/home/steve/Desktop/train-52x52/";
	//const char* path = "/home/steve/Desktop/train-52x52-small/";
	nftw(path, collect_files, 15, FTW_PHYS);
 	cout << "BMP files found: " << files.size() << endl;

 	MatrixXf samplesX;
 	MatrixXf samplesY;
 	BMPFilesToSamples(files, 52*52*3, 12, &samplesX, &samplesY);

	//cout << (samplesX) << endl << endl;
	//cout << (samplesY) << endl << endl;

 	/*Eigen::MatrixXf m(9,2);
 	m << 1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19;

 	ConvolutionalLayer cl(3,3,1,1,0,2,2);
 	cout << cl.forward(m) << endl;*/





 	NN nn(samplesX.rows());
	/*nn.addConvLayer(52,52,3,1,2,3,8);
	/*nn.addConvLayer(2,1,3,64);
	nn.addConvLayer(2,1,3,128);
	nn.addFCLayer(1000);*/
	nn.addFCLayer(1000);
	nn.addFCLayer(samplesY.rows(),true);


	cout << "NN construction completed." << endl;

	/*cout << nn.forward(samplesX.block(0,0,3,samplesX.cols())) << endl;*/

	nn.train(samplesX, samplesY, 10, 0.001f, 0.95f, 250, false);

	ofstream f;
	f.open("nn.dat", std::ofstream::out | std::ofstream::binary);
	nn.save(f);
	f.close();

	return 0;
}
