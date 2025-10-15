#ifndef _MODELS_
#define _MODELS_

#include <string>
#include <vector>
#include <assert.h>
#include "puma.h"
#include "fully-connected-layer.h"
#include "lstm-layer.h"
#include "conv-layer.h"



void simple(Model& model){
    unsigned int size = 5;
    auto in = InputVector::create(model, "in", size);
    ConstantMatrix matrix = ConstantMatrix::create(model, "constant_", size, size);
    OutputVector out = OutputVector::create(model, "out_", size);

    Vector result = matrix * in;
    out = result;
}



void conv_layer(Model& model){
    // Process parameters
    unsigned int in_size_x = 14;
    unsigned int in_size_y = 14;
    unsigned int in_channels = 512;
    unsigned int out_channels = 512;
    unsigned int k_size_x = 3;
    unsigned int k_size_y = 3;
 

    // Input stream
    auto in_stream = InputImagePixelStream::create(model, "in_stream", in_size_x, in_size_y, in_channels);

    // Output stream
    unsigned int out_size_x = in_size_x;
    unsigned int out_size_y = in_size_y;
    auto out_stream = OutputImagePixelStream::create(model, "out_stream", out_size_x, out_size_y, out_channels);

    // Layer
    out_stream = conv_layer(model, "", k_size_x, k_size_y, in_size_x, in_size_y, in_channels, out_channels, in_stream);

}

void convmax(Model& model){
    // Process parameters
    unsigned int in_size_x = 14;
    unsigned int in_size_y = 14;
    unsigned int in_channels = 512;
    unsigned int out_channels = 512;
    unsigned int k_size_x = 3;
    unsigned int k_size_y = 3;
    unsigned int max_pool_size_x = 2;
    unsigned int max_pool_size_y = 2;


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

}


void mlp_l4(Model& model) {
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
 
    auto in = InputVector::create(model, "in" + std::to_string(model_count), in_size);

    auto out = OutputVector::create(model, "out" + std::to_string(model_count++), out_size);

    // Define network
    auto out1 = fully_connected_layer(model, "layer" + std::to_string(layer_count++), in_size1, out_size1, in);
    auto out2 = fully_connected_layer(model, "layer" + std::to_string(layer_count++), in_size2, out_size2, out1);
    auto out3 = fully_connected_layer(model, "layer" + std::to_string(layer_count++), in_size3, out_size3, out2);
    auto out4 = fully_connected_layer(model, "layer" + std::to_string(layer_count++), in_size4, out_size4, out3);
    out = out4;

}



void mlp_l5(Model& model){
    // Input
    unsigned int in_size = 1024;
    unsigned int in_size1 = in_size;
    unsigned int out_size1 = 2048;
    unsigned int in_size2 = out_size1;
    unsigned int out_size2 = 3072;

    // Layer 3 configurations
    unsigned int in_size3 = out_size2;
    unsigned int out_size3 = 3072;

    // Layer 4 configurations
    unsigned int in_size4 = out_size3;
    unsigned int out_size4 = 1024;

    // Layer 5 configurations
    unsigned int in_size5 = out_size4;
    unsigned int out_size5 = 10;

    // Output
    unsigned int out_size = out_size5;



    auto in = InputVector::create(model, "in", in_size);

    // Layer 1 configurations
   

    // Layer 2 configurations
  
    auto out = OutputVector::create(model, "out", out_size);

    // Define network
    auto out1 = fully_connected_layer(model, "layer" + std::to_string(1), in_size1, out_size1, in);
    auto out2 = fully_connected_layer(model, "layer" + std::to_string(2), in_size2, out_size2, out1);
    auto out3 = fully_connected_layer(model, "layer" + std::to_string(3), in_size3, out_size3, out2);
    auto out4 = fully_connected_layer(model, "layer" + std::to_string(4), in_size4, out_size4, out3);
    auto out5 = fully_connected_layer(model, "layer" + std::to_string(5), in_size5, out_size5, out4);
    out = out5;

}


void lstm(Model& model) {

    // Process parameters ??
    unsigned int in_size = 1024;
    unsigned int h_size = 1024;
    unsigned int out_size = 1024;


    // Input
    auto in = InputVector::create(model, "in", in_size);

    // Output
    auto out = OutputVector::create(model, "out", out_size);

    // Layer
    out = lstm_layer(model, "", in_size, h_size, out_size, in);

}

void nmt_l3(Model& model){
        unsigned int in_size = 1024;
    auto in = InputVector::create(model, "in", in_size);

    // Layer 1 configurations
    unsigned int in_size1 = in_size;
    unsigned int h_size1 = 1024;
    unsigned int out_size1 = 1024;

    // Layer 2 configurations
    unsigned int in_size2 = out_size1;
    unsigned int h_size2 = 1024;
    unsigned int out_size2 = 1024;

    // Layer 3 configurations
    unsigned int in_size3 = out_size2;
    unsigned int h_size3 = 1024;
    unsigned int out_size3 = 1024;

    // Layer 4 configurations
    unsigned int in_size4 = out_size3;
    unsigned int h_size4 = 1024;
    unsigned int out_size4 = 1024;

    // Layer 5 configurations
    unsigned int in_size5 = out_size4;
    unsigned int h_size5 = 1024;
    unsigned int out_size5 = 1024;

    // Layer 6 configurations
    unsigned int in_size6 = out_size5;
    unsigned int h_size6 = 1024;
    unsigned int out_size6 = 1024;

    // Layer 7 (linear layer) configurations
    unsigned int in_size7 = out_size6;
    unsigned int out_size7 = 40000;

    // Output
    unsigned int out_size = out_size7;
    auto out = OutputVector::create(model, "out", out_size);

    // Define network
    auto out1 = lstm_layer(model, "layer" + std::to_string(1), in_size1, h_size1, out_size1, in);
    auto out2 = lstm_layer(model, "layer" + std::to_string(2), in_size2, h_size2, out_size2, out1);
    auto out3 = lstm_layer(model, "layer" + std::to_string(3), in_size3, h_size3, out_size3, out2);
    auto out4 = lstm_layer(model, "layer" + std::to_string(4), in_size4, h_size4, out_size4, out3);
    auto out5 = lstm_layer(model, "layer" + std::to_string(5), in_size5, h_size5, out_size5, out4);
    auto out6 = lstm_layer(model, "layer" + std::to_string(6), in_size6, h_size6, out_size6, out5);
    auto out7 = fully_connected_layer(model, "layer" + std::to_string(7), in_size7, out_size7, out6);
    out = out7;
}




void fully_connected_layer(Model& model){
    // Process parameters
    unsigned int in_size = 1024;
    unsigned int out_size = 1024;
  

    // Input
    auto in = InputVector::create(model, "in", in_size);

    // Output
    auto out = OutputVector::create(model, "out", out_size);

    // Layer
    out = fully_connected_layer(model, "", in_size, out_size, in);

}



void nmt_l5(Model& model){
        // Input
    unsigned int in_size = 1024;
    auto in = InputVector::create(model, "in", in_size);

    // Layer 1 configurations
    unsigned int in_size1 = in_size;
    unsigned int h_size1 = 1024;
    unsigned int out_size1 = 1024;

    // Layer 2 configurations
    unsigned int in_size2 = out_size1;
    unsigned int h_size2 = 1024;
    unsigned int out_size2 = 1024;

    // Layer 3 configurations
    unsigned int in_size3 = out_size2;
    unsigned int h_size3 = 1024;
    unsigned int out_size3 = 1024;

    // Layer 4 configurations
    unsigned int in_size4 = out_size3;
    unsigned int h_size4 = 1024;
    unsigned int out_size4 = 1024;

    // Layer 5 configurations
    unsigned int in_size5 = out_size4;
    unsigned int h_size5 = 1024;
    unsigned int out_size5 = 1024;

    // Layer 6 configurations
    unsigned int in_size6 = out_size5;
    unsigned int h_size6 = 1024;
    unsigned int out_size6 = 1024;

    // Layer 7 configurations
    unsigned int in_size7 = out_size6;
    unsigned int h_size7 = 1024;
    unsigned int out_size7 = 1024;

    // Layer 8 configurations
    unsigned int in_size8 = out_size7;
    unsigned int h_size8 = 1024;
    unsigned int out_size8 = 1024;

    // Layer 9 configurations
    unsigned int in_size9 = out_size8;
    unsigned int h_size9 = 1024;
    unsigned int out_size9 = 1024;

    // Layer 10 configurations
    unsigned int in_size10 = out_size9;
    unsigned int h_size10 = 1024;
    unsigned int out_size10 = 1024;

    // Layer 11 (linear layer) configurations
    unsigned int in_size11 = out_size10;
    unsigned int out_size11 = 40000;

    // Output
    unsigned int out_size = out_size11;
    auto out = OutputVector::create(model, "out", out_size);

    // Define network
    auto out1 = lstm_layer(model, "layer" + std::to_string(1), in_size1, h_size1, out_size1, in);
    auto out2 = lstm_layer(model, "layer" + std::to_string(2), in_size2, h_size2, out_size2, out1);
    auto out3 = lstm_layer(model, "layer" + std::to_string(3), in_size3, h_size3, out_size3, out2);
    auto out4 = lstm_layer(model, "layer" + std::to_string(4), in_size4, h_size4, out_size4, out3);
    auto out5 = lstm_layer(model, "layer" + std::to_string(5), in_size5, h_size5, out_size5, out4);
    auto out6 = lstm_layer(model, "layer" + std::to_string(6), in_size6, h_size6, out_size6, out5);
    auto out7 = lstm_layer(model, "layer" + std::to_string(7), in_size7, h_size7, out_size7, out6);
    auto out8 = lstm_layer(model, "layer" + std::to_string(8), in_size8, h_size8, out_size8, out7);
    auto out9 = lstm_layer(model, "layer" + std::to_string(9), in_size9, h_size9, out_size9, out8);
    auto out10 = lstm_layer(model, "layer" + std::to_string(10), in_size10, h_size10, out_size10, out9);
    auto out11 = fully_connected_layer(model, "layer" + std::to_string(11), in_size11, out_size11, out10);
    out = out11;
}


void isolated_fully_connected_layer(Model model, std::string layerName, unsigned int in_size, unsigned int out_size) {

    // Input vector
    auto in = InputVector::create(model, "in", in_size);

    // Output vector
    auto out = OutputVector::create(model, "out", out_size);

    // Layer
    out = fully_connected_layer(model, layerName, in_size, out_size, in);

}

void vgg16(Model& model){
    // Input
    unsigned int in_size_x = 224;
    unsigned int in_size_y = 224;
    unsigned int in_channels = 3;
    auto in_stream = InputImagePixelStream::create(model, "in_stream", in_size_x, in_size_y, in_channels);

    // Layer 1 (convolution) configurations
    unsigned int k_size_x1 = 3;
    unsigned int k_size_y1 = 3;
    unsigned int in_size_x1 = 224;
    unsigned int in_size_y1 = 224;
    unsigned int in_channels1 = 3;
    unsigned int out_channels1 = 64;

    // Layer 2 (convolution with max pool) configurations
    unsigned int k_size_x2 = 3;
    unsigned int k_size_y2 = 3;
    unsigned int in_size_x2 = 224;
    unsigned int in_size_y2 = 224;
    unsigned int in_channels2 = 64;
    unsigned int out_channels2 = 64;
    unsigned int max_pool_size_x2 = 2;
    unsigned int max_pool_size_y2 = 2;

    // Layer 3 (convolution) configurations
    unsigned int k_size_x3 = 3;
    unsigned int k_size_y3 = 3;
    unsigned int in_size_x3 =112;
    unsigned int in_size_y3 = 112;
    unsigned int in_channels3 = 64;
    unsigned int out_channels3 = 128;

    // Layer 4 (convolution with max pool) configurations
    unsigned int k_size_x4 = 3;
    unsigned int k_size_y4 = 3;
    unsigned int in_size_x4 =112;
    unsigned int in_size_y4 = 112;
    unsigned int in_channels4 = 128;
    unsigned int out_channels4 = 128;
    unsigned int max_pool_size_x4 = 2;
    unsigned int max_pool_size_y4 = 2;

    // Layer 5 (convolution) configurations
    unsigned int k_size_x5 = 3;
    unsigned int k_size_y5 = 3;
    unsigned int in_size_x5 =56;
    unsigned int in_size_y5 = 56;
    unsigned int in_channels5 = 128;
    unsigned int out_channels5 = 256;

    // Layer 6 (convolution) configurations
    unsigned int k_size_x6 = 3;
    unsigned int k_size_y6 = 3;
    unsigned int in_size_x6 =56;
    unsigned int in_size_y6 = 56;
    unsigned int in_channels6 = 256;
    unsigned int out_channels6 = 256;

    // Layer 7 (convolution with max pool) configurations
    unsigned int k_size_x7 = 3;
    unsigned int k_size_y7 = 3;
    unsigned int in_size_x7 =56;
    unsigned int in_size_y7 = 56;
    unsigned int in_channels7 = 256;
    unsigned int out_channels7 = 256;
    unsigned int max_pool_size_x7 = 2;
    unsigned int max_pool_size_y7 = 2;

    // Layer 8 (convolution) configurations
    unsigned int k_size_x8 = 3;
    unsigned int k_size_y8 = 3;
    unsigned int in_size_x8 =28;
    unsigned int in_size_y8 = 28;
    unsigned int in_channels8 = 256;
    unsigned int out_channels8 = 512;

    // Layer 9 (convolution) configurations
    unsigned int k_size_x9 = 3;
    unsigned int k_size_y9 = 3;
    unsigned int in_size_x9 =28;
    unsigned int in_size_y9 = 28;
    unsigned int in_channels9 = 512;
    unsigned int out_channels9 = 512;

    // Layer 10 (convolution with max pool) configurations
    unsigned int k_size_x10 = 3;
    unsigned int k_size_y10 = 3;
    unsigned int in_size_x10 =28;
    unsigned int in_size_y10 = 28;
    unsigned int in_channels10 = 512;
    unsigned int out_channels10 = 512;
    unsigned int max_pool_size_x10 = 2;
    unsigned int max_pool_size_y10 = 2;

    // Layer 11 (convolution) configurations
    unsigned int k_size_x11 = 3;
    unsigned int k_size_y11 = 3;
    unsigned int in_size_x11 =14;
    unsigned int in_size_y11 = 14;
    unsigned int in_channels11 = 512;
    unsigned int out_channels11 = 512;

    // Layer 12 (convolution) configurations
    unsigned int k_size_x12 = 3;
    unsigned int k_size_y12 = 3;
    unsigned int in_size_x12 =14;
    unsigned int in_size_y12 = 14;
    unsigned int in_channels12 = 512;
    unsigned int out_channels12 = 512;

    // Layer 13 (convolution with max pool) configurations
    unsigned int k_size_x13 = 3;
    unsigned int k_size_y13 = 3;
    unsigned int in_size_x13 =14;
    unsigned int in_size_y13 = 14;
    unsigned int in_channels13 = 512;
    unsigned int out_channels13 = 512;
    unsigned int max_pool_size_x13 = 2;
    unsigned int max_pool_size_y13 = 2;

    // Output
    unsigned int out_size_x = 7;
    unsigned int out_size_y = 7;
    unsigned int out_channels = 512;
    auto out_stream = OutputImagePixelStream::create(model, "out_stream", out_size_x, out_size_y, out_channels);

    // Layer 14 (fully-connected) configurations
    unsigned int in_size14 = 25088;
    unsigned int out_size14 = 4096;

    // Layer 15 (fully-connected) configurations
    unsigned int in_size15 = 4096;
    unsigned int out_size15 = 4096;

    // Layer 16 (fully-connected) configurations
    unsigned int in_size16 = 4096;
    unsigned int out_size16 = 1000;

    // Define network
    auto out1 = conv_layer(model, "layer" + std::to_string(1), k_size_x1, k_size_y1, in_size_x1, in_size_y1, in_channels1, out_channels1, in_stream);
    auto out2 = convmax_layer(model, "layer" + std::to_string(2), k_size_x2, k_size_y2, in_size_x2, in_size_y2, in_channels2, out_channels2, max_pool_size_x2, max_pool_size_y2, out1);
    auto out3 = conv_layer(model, "layer" + std::to_string(3), k_size_x3, k_size_y3, in_size_x3, in_size_y3, in_channels3, out_channels3, out2);
    auto out4 = convmax_layer(model, "layer" + std::to_string(4), k_size_x4, k_size_y4, in_size_x4, in_size_y4, in_channels4, out_channels4, max_pool_size_x4, max_pool_size_y4, out3);
    auto out5 = conv_layer(model, "layer" + std::to_string(5), k_size_x5, k_size_y5, in_size_x5, in_size_y5, in_channels5, out_channels5, out4);
    auto out6 = conv_layer(model, "layer" + std::to_string(6), k_size_x6, k_size_y6, in_size_x6, in_size_y6, in_channels6, out_channels6, out5);
    auto out7 = convmax_layer(model, "layer" + std::to_string(7), k_size_x7, k_size_y7, in_size_x7, in_size_y7, in_channels7, out_channels7, max_pool_size_x7, max_pool_size_y7, out6);
    auto out8 = conv_layer(model, "layer" + std::to_string(8), k_size_x8, k_size_y8, in_size_x8, in_size_y8, in_channels8, out_channels8, out7);
    auto out9 = conv_layer(model, "layer" + std::to_string(9), k_size_x9, k_size_y9, in_size_x9, in_size_y9, in_channels9, out_channels9, out8);
    auto out10 = convmax_layer(model, "layer" + std::to_string(10), k_size_x10, k_size_y10, in_size_x10, in_size_y10, in_channels10, out_channels10, max_pool_size_x10, max_pool_size_y10, out9);
    auto out11 = conv_layer(model, "layer" + std::to_string(11), k_size_x11, k_size_y11, in_size_x11, in_size_y11, in_channels11, out_channels11, out10);
    auto out12 = conv_layer(model, "layer" + std::to_string(12), k_size_x12, k_size_y12, in_size_x12, in_size_y12, in_channels12, out_channels12, out11);
    auto out13 = convmax_layer(model, "layer" + std::to_string(13), k_size_x13, k_size_y13, in_size_x13, in_size_y13, in_channels13, out_channels13, max_pool_size_x13, max_pool_size_y13, out12);
    out_stream = out13;
    isolated_fully_connected_layer(model, "layer" + std::to_string(14), in_size14, out_size14);
    isolated_fully_connected_layer(model, "layer" + std::to_string(15), in_size15, out_size15);
    isolated_fully_connected_layer(model, "layer" + std::to_string(16), in_size16, out_size16);
}


void vgg19(Model& model){
    // Input
    unsigned int in_size_x = 224;
    unsigned int in_size_y = 224;
    unsigned int in_channels = 3;
    auto in_stream = InputImagePixelStream::create(model, "in_stream", in_size_x, in_size_y, in_channels);

    // Layer 1 (convolution) configurations
    unsigned int k_size_x1 = 3;
    unsigned int k_size_y1 = 3;
    unsigned int in_size_x1 = 224;
    unsigned int in_size_y1 = 224;
    unsigned int in_channels1 = 3;
    unsigned int out_channels1 = 64;

    // Layer 2 (convolution with max pool) configurations
    unsigned int k_size_x2 = 3;
    unsigned int k_size_y2 = 3;
    unsigned int in_size_x2 = 224;
    unsigned int in_size_y2 = 224;
    unsigned int in_channels2 = 64;
    unsigned int out_channels2 = 64;
    unsigned int max_pool_size_x2 = 2;
    unsigned int max_pool_size_y2 = 2;

    // Layer 3 (convolution) configurations
    unsigned int k_size_x3 = 3;
    unsigned int k_size_y3 = 3;
    unsigned int in_size_x3 = 112;
    unsigned int in_size_y3 = 112;
    unsigned int in_channels3 = 64;
    unsigned int out_channels3 = 128;

    // Layer 4 (convolution with max pool) configurations
    unsigned int k_size_x4 = 3;
    unsigned int k_size_y4 = 3;
    unsigned int in_size_x4 = 112;
    unsigned int in_size_y4 = 112;
    unsigned int in_channels4 = 128;
    unsigned int out_channels4 = 128;
    unsigned int max_pool_size_x4 = 2;
    unsigned int max_pool_size_y4 = 2;

    // Layer 5 (convolution) configurations
    unsigned int k_size_x5 = 3;
    unsigned int k_size_y5 = 3;
    unsigned int in_size_x5 = 56;
    unsigned int in_size_y5 = 56;
    unsigned int in_channels5 = 128;
    unsigned int out_channels5 = 256;

    // Layer 6 (convolution) configurations
    unsigned int k_size_x6 = 3;
    unsigned int k_size_y6 = 3;
    unsigned int in_size_x6 = 56;
    unsigned int in_size_y6 = 56;
    unsigned int in_channels6 = 256;
    unsigned int out_channels6 = 256;

    // Layer 6x (convolution) configurations
    unsigned int k_size_x6x = 3;
    unsigned int k_size_y6x = 3;
    unsigned int in_size_x6x = 56;
    unsigned int in_size_y6x = 56;
    unsigned int in_channels6x = 256;
    unsigned int out_channels6x = 256;

    // Layer 7 (convolution with max pool) configurations
    unsigned int k_size_x7 = 3;
    unsigned int k_size_y7 = 3;
    unsigned int in_size_x7 = 56;
    unsigned int in_size_y7 = 56;
    unsigned int in_channels7 = 256;
    unsigned int out_channels7 = 256;
    unsigned int max_pool_size_x7 = 2;
    unsigned int max_pool_size_y7 = 2;

    // Layer 8 (convolution) configurations
    unsigned int k_size_x8 = 3;
    unsigned int k_size_y8 = 3;
    unsigned int in_size_x8 = 28;
    unsigned int in_size_y8 = 28;
    unsigned int in_channels8 = 256;
    unsigned int out_channels8 = 512;

    // Layer 9 (convolution) configurations
    unsigned int k_size_x9 = 3;
    unsigned int k_size_y9 = 3;
    unsigned int in_size_x9 = 28;
    unsigned int in_size_y9 = 28;
    unsigned int in_channels9 = 512;
    unsigned int out_channels9 = 512;

    // Layer 9x (convolution) configurations
    unsigned int k_size_x9x = 3;
    unsigned int k_size_y9x = 3;
    unsigned int in_size_x9x = 28;
    unsigned int in_size_y9x = 28;
    unsigned int in_channels9x = 512;
    unsigned int out_channels9x = 512;

    // Layer 10 (convolution with max pool) configurations
    unsigned int k_size_x10 = 3;
    unsigned int k_size_y10 = 3;
    unsigned int in_size_x10 = 28;
    unsigned int in_size_y10 = 28;
    unsigned int in_channels10 = 512;
    unsigned int out_channels10 = 512;
    unsigned int max_pool_size_x10 = 2;
    unsigned int max_pool_size_y10 = 2;

    // Layer 11 (convolution) configurations
    unsigned int k_size_x11 = 3;
    unsigned int k_size_y11 = 3;
    unsigned int in_size_x11 = 14;
    unsigned int in_size_y11 = 14;
    unsigned int in_channels11 = 512;
    unsigned int out_channels11 = 512;

    // Layer 12 (convolution) configurations
    unsigned int k_size_x12 = 3;
    unsigned int k_size_y12 = 3;
    unsigned int in_size_x12 = 14;
    unsigned int in_size_y12 = 14;
    unsigned int in_channels12 = 512;
    unsigned int out_channels12 = 512;

    // Layer 12x (convolution) configurations
    unsigned int k_size_x12x = 3;
    unsigned int k_size_y12x = 3;
    unsigned int in_size_x12x = 14;
    unsigned int in_size_y12x = 14;
    unsigned int in_channels12x = 512;
    unsigned int out_channels12x = 512;

    // Layer 13 (convolution with max pool) configurations
    unsigned int k_size_x13 = 3;
    unsigned int k_size_y13 = 3;
    unsigned int in_size_x13 = 14;
    unsigned int in_size_y13 = 14;
    unsigned int in_channels13 = 512;
    unsigned int out_channels13 = 512;
    unsigned int max_pool_size_x13 = 2;
    unsigned int max_pool_size_y13 = 2;

    // Output
    unsigned int out_size_x = 7;
    unsigned int out_size_y = 7;
    unsigned int out_channels = 512;
    auto out_stream = OutputImagePixelStream::create(model, "out_stream", out_size_x, out_size_y, out_channels);

    // Layer 17 (fully-connected) configurations
    unsigned int in_size17 = 25088;
    unsigned int out_size17 = 4096;

    // Layer 18 (fully-connected) configurations
    unsigned int in_size18 = 4096;
    unsigned int out_size18 = 4096;

    // Layer 19 (fully-connected) configurations
    unsigned int in_size19 = 4096;
    unsigned int out_size19 = 1000;

    // Define network
    auto out1 = conv_layer(model, "layer" + std::to_string(1), k_size_x1, k_size_y1, in_size_x1, in_size_y1, in_channels1, out_channels1, in_stream);
    auto out2 = convmax_layer(model, "layer" + std::to_string(2), k_size_x2, k_size_y2, in_size_x2, in_size_y2, in_channels2, out_channels2, max_pool_size_x2, max_pool_size_y2, out1);
    auto out3 = conv_layer(model, "layer" + std::to_string(3), k_size_x3, k_size_y3, in_size_x3, in_size_y3, in_channels3, out_channels3, out2);
    auto out4 = convmax_layer(model, "layer" + std::to_string(4), k_size_x4, k_size_y4, in_size_x4, in_size_y4, in_channels4, out_channels4, max_pool_size_x4, max_pool_size_y4, out3);
    auto out5 = conv_layer(model, "layer" + std::to_string(5), k_size_x5, k_size_y5, in_size_x5, in_size_y5, in_channels5, out_channels5, out4);
    auto out6 = conv_layer(model, "layer" + std::to_string(6), k_size_x6, k_size_y6, in_size_x6, in_size_y6, in_channels6, out_channels6, out5);
    auto out6x = conv_layer(model, "layer" + std::to_string(6) + "x", k_size_x6x, k_size_y6x, in_size_x6x, in_size_y6x, in_channels6x, out_channels6x, out6);
    auto out7 = convmax_layer(model, "layer" + std::to_string(7), k_size_x7, k_size_y7, in_size_x7, in_size_y7, in_channels7, out_channels7, max_pool_size_x7, max_pool_size_y7, out6x);
    auto out8 = conv_layer(model, "layer" + std::to_string(8), k_size_x8, k_size_y8, in_size_x8, in_size_y8, in_channels8, out_channels8, out7);
    auto out9 = conv_layer(model, "layer" + std::to_string(9), k_size_x9, k_size_y9, in_size_x9, in_size_y9, in_channels9, out_channels9, out8);
    auto out9x = conv_layer(model, "layer" + std::to_string(9) + "x", k_size_x9x, k_size_y9x, in_size_x9x, in_size_y9x, in_channels9x, out_channels9x, out9);
    auto out10 = convmax_layer(model, "layer" + std::to_string(10), k_size_x10, k_size_y10, in_size_x10, in_size_y10, in_channels10, out_channels10, max_pool_size_x10, max_pool_size_y10, out9x);
    auto out11 = conv_layer(model, "layer" + std::to_string(11), k_size_x11, k_size_y11, in_size_x11, in_size_y11, in_channels11, out_channels11, out10);
    auto out12 = conv_layer(model, "layer" + std::to_string(12), k_size_x12, k_size_y12, in_size_x12, in_size_y12, in_channels12, out_channels12, out11);
    auto out12x = conv_layer(model, "layer" + std::to_string(12) + "x", k_size_x12x, k_size_y12x, in_size_x12x, in_size_y12x, in_channels12x, out_channels12x, out12);
    auto out13 = convmax_layer(model, "layer" + std::to_string(13), k_size_x13, k_size_y13, in_size_x13, in_size_y13, in_channels13, out_channels13, max_pool_size_x13, max_pool_size_y13, out12x);
    out_stream = out13;
    isolated_fully_connected_layer(model, std::to_string(17), in_size17, out_size17);
    isolated_fully_connected_layer(model, std::to_string(18), in_size18, out_size18);
    isolated_fully_connected_layer(model, std::to_string(19), in_size19, out_size19);
}


void wlm_bigLSTM(Model& model){
        // Input
    unsigned int in_size = 8192;
    auto in = InputVector::create(model, "in", in_size);

    // Layer 1 configurations
    unsigned int in_size1 = in_size;
    unsigned int h_size1 = 8192;
    unsigned int out_size1 = 8192;

    // Layer 2 (linear layer) configurations
    unsigned int in_size2 = out_size1;
    unsigned int out_size2 = 1024;

    // Layer 3 configurations
    unsigned int in_size3 = out_size1;
    unsigned int h_size3 = 8192;
    unsigned int out_size3 = 8192;

    // Layer 4 (linear layer) configurations
    unsigned int in_size4 = out_size3;
    unsigned int out_size4 = 1024;

    // Output
    unsigned int out_sizeA = out_size2;
    unsigned int out_sizeB = out_size4;
    auto outA = OutputVector::create(model, "outA", out_sizeA);
    auto outB = OutputVector::create(model, "outB", out_sizeA);

    // Define network
    auto out1 = lstm_layer(model, "layer" + std::to_string(1), in_size1, h_size1, out_size1, in);
    auto out2 = fully_connected_layer(model, "layer" + std::to_string(2), in_size2, out_size2, out1);
    outA = out2;
    auto out3 = lstm_layer(model, "layer" + std::to_string(3), in_size3, h_size3, out_size3, out1);
    auto out4 = fully_connected_layer(model, "layer" + std::to_string(4), in_size4, out_size4, out3);
    outB = out4;
}



void wlm_LSTM2048(Model& model){
       // Input
    unsigned int in_size = 8192;
    auto in = InputVector::create(model, "in", in_size);

    // Layer 1 configurations
    unsigned int in_size1 = in_size;
    unsigned int h_size1 = 8192;
    unsigned int out_size1 = 8192;

    // Layer 2 (linear layer) configurations
    unsigned int in_size2 = out_size1;
    unsigned int out_size2 = 2048;

    // Output
    unsigned int out_size = out_size2;
    auto out = OutputVector::create(model, "out", out_size);

    // Define network
    auto out1 = lstm_layer(model, "layer" + std::to_string(1), in_size1, h_size1, out_size1, in);
    auto out2 = fully_connected_layer(model, "layer" + std::to_string(2), in_size2, out_size2, out1);
    out = out2;
}
#endif
