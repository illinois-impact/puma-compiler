/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>
#include <string>
#include <vector>

#include "puma.h"
#include "conv-layer.h"
#include "fully-connected-layer.h"

void isolated_fully_connected_layer(Model model, std::string layerName, unsigned int in_size, unsigned int out_size) {

    // Input vector
    auto in = InputVector::create(model, "in", in_size);

    // Output vector
    auto out = OutputVector::create(model, "out", out_size);

    // Layer
    out = fully_connected_layer(model, layerName, in_size, out_size, in);

}

int main() {

    Model model = Model::create("lenet");

    // Layer 1 (convolution) configurations
    unsigned int k_size_x1 = 5;
    unsigned int k_size_y1 = 5;
    unsigned int in_size_x1 = 32;
    unsigned int in_size_y1 = 32;
    unsigned int in_channels1 = 1;
    unsigned int out_channels1 = 6;
    unsigned int max_pool_size_x1 = 2;
    unsigned int max_pool_size_y1 = 2;

    // Layer 2 (convolution) configurations
    unsigned int k_size_x2 = 5;
    unsigned int k_size_y2 = 5;
    unsigned int in_size_x2 = in_size_x1/2; // 16
    unsigned int in_size_y2 = in_size_y1/2;
    unsigned int in_channels2 = out_channels1;
    unsigned int out_channels2 = 16;
    unsigned int max_pool_size_x2 = 2;
    unsigned int max_pool_size_y2 = 2;

    // Layer 3 (convolution) configurations
    unsigned int k_size_x3 = 5;
    unsigned int k_size_y3 = 5;
    unsigned int in_size_x3 = in_size_x2/2; // 8
    unsigned int in_size_y3 = in_size_y2/2;
    unsigned int in_channels3 = out_channels2;
    unsigned int out_channels3 = 32;
    
    // Input
    unsigned int in_size_x = in_size_x2;
    unsigned int in_size_y = in_size_y2;
    unsigned int in_channels = in_channels2;
    auto in_stream = InputImagePixelStream::create(model, "in_stream", in_size_x, in_size_y, in_channels);

    // Output
    unsigned int out_size_x = in_size_x3; // 8
    unsigned int out_size_y = in_size_y3; // 8
    unsigned int out_channels = out_channels2;
    auto out_stream = OutputImagePixelStream::create(model, "out_stream", out_size_x, out_size_y, out_channels);

    // Layer 4 (fully-connected) configurations
    unsigned int in_size4 = 512;
    unsigned int out_size4 = 10;

    // Define network
    // auto out1 = convmax_layer(model, "layer" + std::to_string(1), k_size_x1, k_size_y1, in_size_x1, in_size_y1, in_channels1, out_channels1, max_pool_size_x1, max_pool_size_y1, in_stream);
    // auto out1 = conv_layer(model, "layer" + std::to_string(1), k_size_x1, k_size_y1, in_size_x1, in_size_y1, in_channels1, out_channels1, in_stream);
    auto out2 = convmax_layer(model, "layer" + std::to_string(2), k_size_x2, k_size_y2, in_size_x2, in_size_y2, in_channels2, out_channels2, max_pool_size_x2, max_pool_size_y2, in_stream);
    // auto out3 = conv_layer(model, "layer" + std::to_string(3), k_size_x3, k_size_y3, in_size_x3, in_size_y3, in_channels3, out_channels3, out2);
    // out_stream = out3;
    out_stream = out2;
    // isolated_fully_connected_layer(model, "layer" + std::to_string(4), in_size4, out_size4);

    // Compile
    model.compile();

    // Destroy model
    model.destroy();

    return 0;

}

