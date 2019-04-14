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

int main(int argc, char** argv) {

    Model model = Model::create("convmax-layer");

    // Process parameters
    unsigned int in_size_x = 14;
    unsigned int in_size_y = 14;
    unsigned int in_channels = 512;
    unsigned int out_channels = 512;
    unsigned int k_size_x = 3;
    unsigned int k_size_y = 3;
    unsigned int max_pool_size_x = 2;
    unsigned int max_pool_size_y = 2;
    if(argc == 9) {
        in_size_x = atoi(argv[1]);
        in_size_y = atoi(argv[2]);
        in_channels = atoi(argv[3]);
        out_channels = atoi(argv[4]);
        k_size_x = atoi(argv[5]);
        k_size_y = atoi(argv[6]);
        max_pool_size_x = atoi(argv[7]);
        max_pool_size_y = atoi(argv[8]);
    }

    // Input stream
    auto in_stream = InputImagePixelStream::create(model, "in_stream", in_size_x, in_size_y, in_channels);

    // Layer configurations

    // Output stream
    unsigned int conv_out_size_x = in_size_x;
    unsigned int conv_out_size_y = in_size_y;
    unsigned int out_size_x = (conv_out_size_x - 1)/max_pool_size_x + 1;
    unsigned int out_size_y = (conv_out_size_y - 1)/max_pool_size_y + 1;
    auto out_stream = OutputImagePixelStream::create(model, "out_stream", out_size_x, out_size_y, out_channels);

    // Layer
    out_stream = convmax_layer(model, "", k_size_x, k_size_y, in_size_x, in_size_y, in_channels, out_channels, max_pool_size_x, max_pool_size_y, in_stream);

    // Compile
    model.compile();

    // Destroy model
    model.destroy();

    return 0;

}

