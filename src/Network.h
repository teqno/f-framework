#pragma once

#include "Layer.h"

class Network
{
private:
    std::vector<Layer*> layers;
public:
    Network(std::vector<Layer*> &layers);
};