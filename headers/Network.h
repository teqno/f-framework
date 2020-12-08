#pragma once

#include "Layer.h"
#include "loss.h"
#include "Cache.h"

class Network
{
private:
    std::vector<Layer *> layers;
    std::vector<Eigen::VectorXd> activations;
    std::vector<Eigen::VectorXd> preactivations;
    Eigen::MatrixXd retainedGradient;

public:
    Network(std::vector<Layer *> &layers);
    
    std::vector<Layer *> getLayers();
    
    Eigen::MatrixXd forward_prop(const Eigen::MatrixXd &input);
    
    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> back_prop(Eigen::MatrixXd y);
    
    void updateParameters(std::vector<Eigen::MatrixXd> dw, std::vector<Eigen::MatrixXd> db, double alpha, double alphaMomentum);
    
    double calc_cost(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y);
    
    Eigen::VectorXd train(Eigen::MatrixXd &x, Eigen::MatrixXd &y, int epochs, double alpha, double alphaMomentum);
};