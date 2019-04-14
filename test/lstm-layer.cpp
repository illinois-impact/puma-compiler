/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <string>
#include <vector>

#include "puma.h"
#include "lstm-layer.h"

int main(int argc, char** argv) {

    Model model = Model::create("lstm-layer");

    // Process parameters
    unsigned int in_size = 1024;
    unsigned int h_size = 1024;
    unsigned int out_size = 1024;
    if(argc == 4) {
        in_size = atoi(argv[1]);
        h_size = atoi(argv[2]);
        out_size = atoi(argv[2]);
    }

    // Input
    auto in = InputVector::create(model, "in", in_size);

    // Output
    auto out = OutputVector::create(model, "out", out_size);

    // Layer
    out = lstm_layer(model, "", in_size, h_size, out_size, in);

    // Compile
    model.compile();

    // Destroy model
    model.destroy();

    return 0;

}

