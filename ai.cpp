#include <Eigen/Dense>
#include <iostream>
#include <vector>

using namespace std;
using namespace Eigen;

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        this->outputSize = outputSize;

        weights1 = MatrixXd::Random(inputSize, hiddenSize);
        weights2 = MatrixXd::Random(hiddenSize, outputSize);
    }

    MatrixXd sigmoid(const MatrixXd& x) {
        return 1.0 / (1.0 + (-x).array().exp());
    }

    MatrixXd sigmoidDerivative(const MatrixXd& x) {
        return x.array() * (1 - x.array());
    }

    MatrixXd forward(const MatrixXd& X) {
        this->layer1 = sigmoid(X * weights1);
        this->output = sigmoid(layer1 * weights2);
        return output;
    }

    void backward(const MatrixXd& X, const MatrixXd& y, const MatrixXd& output) {
        MatrixXd d_weights2 = layer1.transpose() * (2 * (y - output) * sigmoidDerivative(output));
        MatrixXd d_weights1 = X.transpose() * ((2 * (y - output) * sigmoidDerivative(output)) * weights2.transpose() * sigmoidDerivative(layer1));

        weights1 += d_weights1;
        weights2 += d_weights2;
    }

    void train(const MatrixXd& X, const MatrixXd& y) {
        MatrixXd output = forward(X);
        backward(X, y, output);
    }

private:
    int inputSize;
    int hiddenSize;
    int outputSize;
    MatrixXd weights1;
    MatrixXd weights2;
    MatrixXd layer1;
    MatrixXd output;
};
