/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _PUMA_TEST_LSTM_LAYER_
#define _PUMA_TEST_LSTM_LAYER_

static Vector lstm_layer(Model model, std::string layerName, unsigned int in_size, unsigned int h_size, unsigned int out_size, Vector in) {

    // Hidden Layer 1 weights (i_h1, h1_h1 - recurrent connection)
    std::vector<ConstantMatrix> M1(4);
    for(unsigned int i = 0; i < 4; ++i) {
        M1[i] = ConstantMatrix::create(model, layerName + "M1_" + std::to_string(i), in_size, h_size);
    }
    std::vector<ConstantMatrix> M2(4);
    for(unsigned int i = 0; i < 4; ++i) {
        M2[i] = ConstantMatrix::create(model, layerName + "M2_" + std::to_string(i), h_size, h_size);
    }

    // Hidden Layer 2 weights (h1_h2, h2_h2 - recurrent connection)
    std::vector<ConstantMatrix> M3(4);
    for(unsigned int i = 0; i < 4; ++i) {
        M3[i] = ConstantMatrix::create(model, layerName + "M3_" + std::to_string(i), h_size, h_size);
    }
    std::vector<ConstantMatrix> M4(4);
    for(unsigned int i = 0; i < 4; ++i) {
        M4[i] = ConstantMatrix::create(model, layerName + "M4_" + std::to_string(i), h_size, h_size);
    }

    // Output Layer weights (h2_out)
    ConstantMatrix M5 = ConstantMatrix::create(model, layerName + "M5", h_size, out_size);

    // These vector will be self-modifying, (output of time-step t-1 is input for time-step t)
    auto h1in = InputVector::create(model, layerName + "h1in", h_size);
    auto c1in = InputVector::create(model, layerName + "c1in", h_size);
    auto h2in = InputVector::create(model, layerName + "h2in", h_size);
    auto c2in = InputVector::create(model, layerName + "c2in", h_size);
    auto h1out = OutputVector::create(model, layerName + "h1out", h_size);
    auto c1out = OutputVector::create(model, layerName + "c1out", h_size);
    auto h2out = OutputVector::create(model, layerName + "h2out", h_size);
    auto c2out = OutputVector::create(model, layerName + "c2out", h_size);

    // Computing hidden layer 1
    auto preact1_1 = M1[0]*in + M2[0]*h1in;
    auto preact1_2 = M1[1]*in + M2[1]*h1in;
    auto preact1_3 = M1[2]*in + M2[2]*h1in;
    auto preact1_4 = M1[3]*in + M2[3]*h1in;
    auto i_gate1 = sig(preact1_1);
    auto f_gate1 = sig(preact1_2);
    auto o_gate1 = sig(preact1_3);
    auto c_int1 = tanh(preact1_4);
    auto c1 = f_gate1*c1in + i_gate1*c_int1;
    auto h1 = o_gate1*tanh(c1); // * - element2element multiplication when both operands are vectors
    h1out = h1;
    c1out = c1;

    // Computing hidden layer 2
    auto preact2_1 = M3[0]*h1 + M4[0]*h2in;
    auto preact2_2 = M3[1]*h1 + M4[1]*h2in;
    auto preact2_3 = M3[2]*h1 + M4[2]*h2in;
    auto preact2_4 = M3[3]*h1 + M4[3]*h2in;
    auto i_gate2 = sig(preact2_1);
    auto f_gate2 = sig(preact2_2);
    auto o_gate2 = sig(preact2_3);
    auto c_int2 = tanh(preact2_4);
    auto c2 = f_gate2*c2in + i_gate2*c_int2;
    auto h2 = o_gate2*tanh(c2);
    h2out = h2;
    c2out = c2;

    // Computing output layer
    return M5*h2;

}

#endif

