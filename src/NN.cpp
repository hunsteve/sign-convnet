/*
 * NN.cpp
 *
 *  Created on: Nov 11, 2016
 *      Author: steve
 */

#include "NN.h"
#include <math.h>
#include <stddef.h>
#include <iostream>
#include "ConvolutionalLayer.h"
#include "FullyConnectedLayer.h"

NN::NN(int inputSize) : inputSize(inputSize) {}

NN::~NN() {
    for (size_t i = 0; i < layers.size(); ++i) {
        delete layers[i];
    }
}

Eigen::MatrixXf NN::forward(const Eigen::MatrixXf& input) {
    Eigen::MatrixXf layerOutput = input;
    for (size_t i = 0; i < layers.size(); ++i) {
        layerOutput = layers[i]->forward(layerOutput);
    }

    return layerOutput;
}

void NN::calcDeltas(const Eigen::MatrixXf& error) {
    Eigen::MatrixXf backPropError = error;
    for (size_t i = layers.size() - 1; i < layers.size(); --i) {
        backPropError = layers[i]->backprop(backPropError);
    }
}

void NN::applyWeightMod(float mu) {
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i]->applyWeightMod(mu);
    }
}

void NN::train(const Eigen::MatrixXf& trainX, const Eigen::MatrixXf& trainY,
               int maxEpoch, float mu, float ratio, int minibatchSize,
               bool isDebug) {
    int trainBatchCount = (int)floor((trainX.cols() / minibatchSize) * ratio);

    for (int epoch = 0; epoch < maxEpoch; ++epoch) {
        for (int batch = 0; batch < trainBatchCount; ++batch) {
            Eigen::MatrixXf inp = trainX.block(0, batch * minibatchSize,
                                               trainX.rows(), minibatchSize);
            Eigen::MatrixXf targ = trainY.block(0, batch * minibatchSize,
                                                trainY.rows(), minibatchSize);
            Eigen::MatrixXf out = forward(inp);
            Eigen::MatrixXf err = targ - out;
            calcDeltas(err);
            applyWeightMod(2 * mu / minibatchSize);

            float trainErrorSum = err.squaredNorm() / (err.rows() * err.cols());
            std::cout << "epoch: " << epoch << "/" << maxEpoch
                      << " batch: " << batch << "/" << trainBatchCount
                      << " errorTrain: " << trainErrorSum
                      << " accuracy: " << accuracy(out, targ) << std::endl;
        }

        int validationSize = trainX.cols() - trainBatchCount * minibatchSize;
        std::cout << validationSize << std::endl;
        Eigen::MatrixXf inp = trainX.block(0, trainBatchCount * minibatchSize,
                                           trainX.rows(), validationSize);
        Eigen::MatrixXf targ = trainY.block(0, trainBatchCount * minibatchSize,
                                            trainY.rows(), validationSize);
        Eigen::MatrixXf out = forward(inp);
        Eigen::MatrixXf err = targ - out;
        float MSE = err.squaredNorm() / (err.rows() * err.cols());

        std::cout << "MSE:" << MSE << " accuracy: " << accuracy(out, targ)
                  << std::endl;
    }
}

void NN::addConvLayer(int w, int h, int d, int stride, int padding, int K,
                      int N) {
    int lastSize = 0;
    if (!layers.size())
        lastSize = inputSize;
    else
        lastSize = layers.back()->getOutputSize();
    Layer* layer = new ConvolutionalLayer(w, h, d, stride, padding, K, N);
    assert(w * h * d == lastSize);
    layers.push_back(layer);
}

void NN::addConvLayer(int stride, int padding, int K, int N) {
    assert(layers.size());
    ConvolutionalLayer* lastConvLayer =
        dynamic_cast<ConvolutionalLayer*>(layers.back());
    assert(lastConvLayer);

    Layer* layer = new ConvolutionalLayer(
        lastConvLayer->getOutputWidth(), lastConvLayer->getOutputHeight(),
        lastConvLayer->getOutputDimension(), stride, padding, K, N);
    layers.push_back(layer);
}

void NN::addFCLayer(int size, bool isLinear) {
    int lastSize = 0;
    if (layers.empty())
        lastSize = inputSize;
    else
        lastSize = layers.back()->getOutputSize();
    Layer* layer = new FullyConnectedLayer(lastSize, size, isLinear);
    layers.push_back(layer);
}

float NN::accuracy(const Eigen::MatrixXf& output,
                   const Eigen::MatrixXf& target) const {
    int hit = 0;
    for (int i = 0; i < output.cols(); ++i) {
        Eigen::MatrixXf::Index rowindex;
        output.col(i).maxCoeff(&rowindex);
        if (target(rowindex, i) == 1) hit++;
    }

    return (float)hit / output.cols();
}

Eigen::VectorXi NN::classify(const Eigen::MatrixXf& input) {
    Eigen::MatrixXf out = forward(input);

    Eigen::VectorXi idx(input.cols());
    for (int i = 0; i < input.cols(); ++i) {
        Eigen::MatrixXf::Index rowindex;
        out.col(i).maxCoeff(&rowindex);
        idx(i) = rowindex;
    }

    return idx;
}

void NN::save(std::ostream& out) const {
    out.write((char*)(&inputSize), sizeof(int));
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i]->save(out);
    }
}

NN NN::load(std::istream& in) {
    int inputSize;
    in.read((char*)(&inputSize), sizeof(int));
    NN nn(inputSize);
    while (true) {
        char c;
        in.read(&c, sizeof(char));
        if (in.eof()) break;
        Layer* nextLayer;
        if (c == 'F')
            nextLayer = FullyConnectedLayer::load(in);
        else if (c == 'C')
            nextLayer = ConvolutionalLayer::load(in);
        nn.layers.push_back(nextLayer);
    }

    return nn;
}
