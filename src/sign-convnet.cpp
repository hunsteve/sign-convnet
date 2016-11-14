//============================================================================
// Name        : convnet.cpp
// Author      : Istvan Engedy
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
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

#define HEX( x ) setw(2) << setfill('0') << hex << (int)( x )

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

void BMPFilesToSamples(vector<string> files, int cols, int classes, MatrixXf* samplesX, MatrixXf* samplesY) {
	random_shuffle(files.begin(), files.end());


	*samplesX = MatrixXf(files.size(), cols);
	*samplesY = MatrixXf(files.size(), classes);

	int i=0;
	for (vector<string>::iterator it=files.begin(); it!=files.end(); ++it) {

	    vector<unsigned char> data = readBMP(it->c_str());
	    typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix8u;

	    MatrixXf datarow = Map<Matrix8u>(data.data(), 1, data.size()).cast<MatrixXf::Scalar>();

	    datarow = (datarow - MatrixXf::Ones(1,data.size()) * 128) / 128.0f; //normalize to near N(0,1)

	    samplesX->block(i,0,1,cols) = datarow;
	    samplesY->block(i,0,1,classes) = MatrixXf::Zero(1,classes);
	    (*samplesY)(i,filenameToClassIndex(*it)) = 1;
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
	/*Eigen::MatrixXf m(6,3);
	m << 1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19;

	cout << m << endl;

	m.transposeInPlace();
	MatrixXf v(Map<MatrixXf>(m.data(), m.cols()*m.rows() / 2, 2));
	v.transposeInPlace();

	cout << v << endl;
	 */


	//const char* path = "/home/steve/Desktop/train-52x52/";
	const char* path = "/home/steve/Desktop/train-52x52-small/";
	nftw(path, collect_files, 15, FTW_PHYS);
 	cout << "BMP files found: " << files.size() << endl;

 	MatrixXf samplesX;
 	MatrixXf samplesY;
 	BMPFilesToSamples(files, 52*52*3, 12, &samplesX, &samplesY);

	//cout << (samplesX) << endl << endl;
	//cout << (samplesY) << endl << endl;

 	Eigen::MatrixXf m(2,9);
 	m << 1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19;

 	ConvolutionalLayer cl(3,3,1,1,0,2,2);
 	cout << cl.forward(m) << endl;

	NN nn(samplesX.cols());
	nn.addFCLayer(20);
	nn.addFCLayer(samplesY.cols(),true);

	cout << nn.forward(samplesX.block(0,0,3,samplesX.cols())) << endl;

	nn.train(samplesX, samplesY, 10, 0.01f, 0.8f, 128, false);

	return 0;
}
