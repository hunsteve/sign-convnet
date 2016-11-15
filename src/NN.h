/*
 * NN.h
 *
 *  Created on: Nov 11, 2016
 *      Author: steve
 */

#ifndef NN_H_
#define NN_H_
#include <istream>
#include <ostream>
#include <string>
#include <vector>
#include <memory>
#include "Layer.h"

class NN {
   private:
    std::vector<std::unique_ptr<Layer>> layers;
    const int inputSize;

    float accuracy(const Eigen::MatrixXf& output,
                   const Eigen::MatrixXf& target) const;

    NN & operator=(const NN&) = delete;
    NN(const NN&) = delete;
   public:
    NN(int inputSize);
    virtual ~NN();

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    Eigen::VectorXi classify(const Eigen::MatrixXf& input);

    void calcDeltas(const Eigen::MatrixXf& error);
    void applyWeightMod(float mu);

    void train(const Eigen::MatrixXf& trainX, const Eigen::MatrixXf& trainY,
               int maxEpoch, float mu, float ratio, int minibatchSize,
			   void(*epochEndCallback)(NN*, int)=NULL);

    void addConvLayer(int w, int h, int d, int stride, int padding, int K,
                      int N);
    void addConvLayer(int stride, int padding, int K, int N);
    void addFCLayer(int size, bool isLinear = false);

    void save(std::ostream& out) const;
    static std::unique_ptr<NN> load(std::istream& in);
};

#endif /* NN_H_ */
