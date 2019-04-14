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

int main(int argc, char** argv) {

    Model model = Model::create("simple");
    unsigned int size = 5;
    auto in = InputVector::create(model, "in", size);
    ConstantMatrix matrix = ConstantMatrix::create(model, "constant_", size, size);
    OutputVector out = OutputVector::create(model, "out_", size);

    Vector result = matrix * in;
    out = result;

     // Compile
    model.compile();

    // Destroy model
    model.destroy();

    return 0;

}
