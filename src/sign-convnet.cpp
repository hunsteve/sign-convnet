/*
 * sign-convnet.cpp
 *
 *  Created on: Nov 11, 2016
 *      Author: steve
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include "Eigen/Dense"

#include <ftw.h>
#include <algorithm>
#include "ConvolutionalLayer.h"
#include "FullyConnectedLayer.h"
#include "NN.h"
#include "bitmap.h"

std::vector<std::string> files;

int collect_files(const char* filepath, const struct stat* info,
                  const int typeflag, struct FTW* pathinfo) {
    if (typeflag == FTW_F) {
    	std::string fn(filepath);
        if (!fn.compare(fn.length() - 4, 4, ".bmp")) files.push_back(fn);
    }
    return 0;
}

int filenameToClassIndex(std::string fn) {
    unsigned int from = fn.rfind("/") + 1;
    if (from == std::string::npos) from = 0;
    unsigned int till = fn.rfind("_");
    int sampleClass;
    std::stringstream ss(fn.substr(from, till - from));
    ss >> sampleClass;
    sampleClass -= 1;

    return sampleClass;
}

void BMPFilesToSamples(std::vector<std::string> files, int features, int classes,
		               Eigen::MatrixXf* samplesX, Eigen::MatrixXf* samplesY) {
    random_shuffle(files.begin(), files.end());

    *samplesX = Eigen::MatrixXf(features, files.size());
    *samplesY = Eigen::MatrixXf(classes, files.size());

    int i = 0;
    for (std::vector<std::string>::iterator it = files.begin(); it != files.end(); ++it) {
    	std::vector<unsigned char> data = readBMP(it->c_str());
        typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>
            Matrix8u;

        Eigen::MatrixXf datacol =
        		Eigen::Map<Matrix8u>(data.data(), data.size(), 1).cast<Eigen::MatrixXf::Scalar>();

        datacol = (datacol - Eigen::MatrixXf::Ones(data.size(), 1) * 128) /
                  128.0f;  // normalize to near N(0,1)

        samplesX->block(0, i, features, 1) = datacol;
        samplesY->block(0, i, classes, 1) = Eigen::MatrixXf::Zero(classes, 1);
        (*samplesY)(filenameToClassIndex(*it), i) = 1;
        i++;
        if (i % 1000 == 0) std::cout << "Read " << i << " files." << std::endl;
    }
}

void saveNNCallback(NN* nn, int epoch) {
	std::stringstream ss;
	ss << "nn" << epoch << ".dat";
	std::ofstream f;
	f.open(ss.str(), std::ofstream::out | std::ofstream::binary);
	nn->save(f);
	f.close();
}

void perform_gradient_check() {
	NN nn(20);
	nn.addFCLayer(50);
	nn.addFCLayer(50);
	nn.addFCLayer(3, true);
	std::cout << "Gradient check on pure FC net, largest deviation: " << nn.gradientCheck() << std::endl;

	NN nn2(300);
	nn2.addConvLayer(10,10,3,1,2,3,5);
	nn2.addConvLayer(2,1,3,10);
	nn2.addFCLayer(50);
	nn2.addFCLayer(3, true);
	std::cout << "Gradient check on convnet, largest deviation: " << nn2.gradientCheck() << std::endl;
}

int main(int argc, char** argv) {

	//if a command line argument is provided, it is considered as an input file
	//it is then classified, using nn.dat, classes: 0..11
	if (argc > 1) {
		char* fn = argv[1];
		std::vector<unsigned char> data = readBMP(fn);
		std::cout << fn << std::endl;
		typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic,
									  Eigen::RowMajor>
			Matrix8u;
		Eigen::MatrixXf samplesX(data.size(), 1);
		Eigen::MatrixXf datacol =
					Eigen::Map<Matrix8u>(data.data(), data.size(), 1).cast<Eigen::MatrixXf::Scalar>();

		datacol = (datacol - Eigen::MatrixXf::Ones(data.size(), 1) * 128) /
					128.0f;  // normalize to near N(0,1)

		samplesX.block(0, 0, data.size(), 1) = datacol;

		std::ifstream f2;
		f2.open("nn.dat", std::ifstream::in | std::ofstream::binary);
		std::unique_ptr<NN> nn = NN::load(f2);
		f2.close();

		std::cout << nn->classify(samplesX) << std::endl;
		return 0;
	}

    Eigen::initParallel();

    std::cout << "Threads used: " << Eigen::nbThreads() << std::endl;
    if (Eigen::nbThreads() == 1)
    	std::cout << "Warning! using only one thread! make sure to use -fopenmp "
                "with G++"
             << std::endl;

    perform_gradient_check();

    const char* path = "/home/steve/Desktop/train-52x52/";
    // const char* path = "/home/steve/Desktop/train-52x52-small/";
    nftw(path, collect_files, 15, FTW_PHYS);
    std::cout << "BMP files found: " << files.size() << std::endl;

    Eigen::MatrixXf samplesX;
    Eigen::MatrixXf samplesY;
    BMPFilesToSamples(files, 52 * 52 * 3, 12, &samplesX, &samplesY);

    //neural net #1
    NN nn(samplesX.rows());
    nn.addConvLayer(52,52,3,1,2,3,32);
    nn.addConvLayer(2,1,3,64);
    nn.addConvLayer(2,1,3,128);
    nn.addFCLayer(1000);
    nn.addFCLayer(samplesY.rows(), true);

    //neural net #2
    /*NN nn(samplesX.rows());
	nn.addConvLayer(52,52,3,1,2,3,64);
	nn.addFCLayer(samplesY.rows(), true);*/

    std::cout << "NN construction completed." << std::endl;

    nn.train(samplesX, samplesY, 5, 0.01f, 0.95f, 125, saveNNCallback);

    return 0;
}
