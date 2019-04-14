/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "puma.h"
#include "fully-connected-layer.h"

int main(int argc, char** argv) {

    Model model = Model::create("fully-connected-layer");

    // Process parameters
    unsigned int in_size = 1024;
    unsigned int out_size = 1024;
    if(argc == 3) {
        in_size = atoi(argv[1]);
        out_size = atoi(argv[2]);
    }

    // Input
    auto in = InputVector::create(model, "in", in_size);

    // Output
    auto out = OutputVector::create(model, "out", out_size);

    // Layer
    out = fully_connected_layer(model, "", in_size, out_size, in);

    // Compile
    model.compile();

    // Bind data
    ModelInstance modelInstance = ModelInstance::create(model);
    float* weights = new float[in_size*out_size];
    fully_connected_layer_bind(modelInstance, "", weights);
    modelInstance.generateData();

    // Destroy model
    model.destroy();
    delete[] weights;

    return 0;

}

