/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "puma.h"
#include "fully-connected-layer.h"

// Input
unsigned int in_size = 128;

// Layer 1 configurations
unsigned int in_size1 = in_size;
unsigned int out_size1 = 256;

// Layer 2 configurations
unsigned int in_size2 = out_size1;
unsigned int out_size2 = 512;

// Layer 3 configurations
unsigned int in_size3 = out_size2;
unsigned int out_size3 = 256;

// Layer 4 configurations
unsigned int in_size4 = out_size3;
unsigned int out_size4 = 10;

// Output
unsigned int out_size = out_size4;

unsigned int layer_count = 0;
unsigned int model_count = 0;

void mlp_l4(Model& model) {
 
    auto in = InputVector::create(model, "in" + std::to_string(model_count), in_size);

    auto out = OutputVector::create(model, "out" + std::to_string(model_count++), out_size);

    // Define network
    auto out1 = fully_connected_layer(model, "layer" + std::to_string(layer_count++), in_size1, out_size1, in);
    auto out2 = fully_connected_layer(model, "layer" + std::to_string(layer_count++), in_size2, out_size2, out1);
    auto out3 = fully_connected_layer(model, "layer" + std::to_string(layer_count++), in_size3, out_size3, out2);
    auto out4 = fully_connected_layer(model, "layer" + std::to_string(layer_count++), in_size4, out_size4, out3);
    out = out4;

}

int main() {

    Model model = Model::create("mlp-l4");

    mlp_l4(model);
    mlp_l4(model); 

    // Compile
    model.compile();

    // Destroy model
    model.destroy();

    return 0;

}
