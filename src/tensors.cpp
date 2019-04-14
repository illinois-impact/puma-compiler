/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>
#include <sstream>

#include "model.h"
#include "operations.h"
#include "tensors.h"

InputVector InputVector::create(Model model, std::string name, unsigned int length) {
    InputVector vec;
    vec.impl_ = new InputVectorImpl(model.unwrap(), name, length);
    return vec;
}

InputImagePixelStream InputImagePixelStream::create(Model model, std::string name, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels) {
    InputImagePixelStream stream;
    stream.impl_ = new InputImagePixelStreamImpl(model.unwrap(), name, imageWidth, imageHeight, nChannels);
    return stream;
}

OutputVector OutputVector::create(Model model, std::string name, unsigned int length) {
    OutputVector vec;
    vec.impl_ = new OutputVectorImpl(model.unwrap(), name, length);
    return vec;
}

OutputImagePixelStream OutputImagePixelStream::create(Model model, std::string name, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels) {
    OutputImagePixelStream stream;
    stream.impl_ = new OutputImagePixelStreamImpl(model.unwrap(), name, imageWidth, imageHeight, nChannels);
    return stream;
}

ConstantMatrix ConstantMatrix::create(Model model, std::string name, unsigned int width, unsigned int height) {
    ConstantMatrix m;
    m.impl_ = new ConstantMatrixImpl(model.unwrap(), name, width, height);
    return m;
}

ConvolutionalConstantMatrix ConvolutionalConstantMatrix::create(Model model, std::string name, unsigned int kernelWidth, unsigned int kernelHeight, unsigned int nInChannels, unsigned int nOutChannels) {
    ConvolutionalConstantMatrix m;
    m.impl_ = new ConvolutionalConstantMatrixImpl(model.unwrap(), name, kernelWidth, kernelHeight, nInChannels, nOutChannels);
    return m;
}

TrainingMatrix TrainingMatrix::create(Model model, std::string name, unsigned int width, unsigned int height) {
    TrainingMatrix m;
    m.impl_ = new TrainingMatrixImpl(model.unwrap(), name, width, height);
    return m;
}

Vector::Vector(VectorImpl* impl)
    : impl_(impl)
{ }

ImagePixelStream::ImagePixelStream(ImagePixelStreamImpl* impl)
    : impl_(impl)
{ }

Transpose::Transpose(TrainingMatrix m)
    : m_(m.unwrap())
{ }

OuterProduct::OuterProduct(Vector x1, Vector x2)
    : x1_(x1.unwrap()), x2_(x2.unwrap())
{ }

InputVectorImpl* InputVector::unwrap() {
    return impl_;
}

InputImagePixelStreamImpl* InputImagePixelStream::unwrap() {
    return impl_;
}

OutputVectorImpl* OutputVector::unwrap() {
    return impl_;
}

OutputImagePixelStreamImpl* OutputImagePixelStream::unwrap() {
    return impl_;
}

VectorImpl* Vector::unwrap() {
    return impl_;
}

ImagePixelStreamImpl* ImagePixelStream::unwrap() {
    return impl_;
}

ConstantMatrixImpl* ConstantMatrix::unwrap() {
    return impl_;
}

ConvolutionalConstantMatrixImpl* ConvolutionalConstantMatrix::unwrap() {
    return impl_;
}

TrainingMatrixImpl* TrainingMatrix::unwrap() {
    return impl_;
}

TrainingMatrixImpl* Transpose::unwrap() {
    return m_;
}

VectorImpl* OuterProduct::unwrap1() {
    return x1_;
}

VectorImpl* OuterProduct::unwrap2() {
    return x2_;
}

InputVectorImpl::InputVectorImpl(ModelImpl* model, std::string name, unsigned int length)
    : AbstractVector(model, name, length)
{
    tiles_.resize(nTiles());
    for(unsigned int i = 0; i < nTiles(); ++i) {
        unsigned int tileSize = MVMU_DIM;
        if(i == nTiles() - 1 && length%MVMU_DIM > 0) {
            tileSize = length%MVMU_DIM;
        }
        tiles_[i] = new InputVectorTile(model, name + "[" + std::to_string(i) + "]", tileSize);
    }
    model->addInputVectorImpl(this);
}

InputImagePixelStreamTile::InputImagePixelStreamTile(ModelImpl* model, std::string name, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels)
    : AbstractImagePixelStream(model, name, imageWidth, imageHeight, nChannels)
{
    stream_.resize(imageHeight);
    for(unsigned int h = 0; h < imageHeight; ++h) {
        stream_[h].resize(imageWidth);
        for(unsigned int w = 0; w < imageWidth; ++w) {
            stream_[h][w] = new InputVectorTile(model, name + "[" + std::to_string(h) + "][" + std::to_string(w) + "]", nChannels);
        }
    }
}

InputImagePixelStreamImpl::InputImagePixelStreamImpl(ModelImpl* model, std::string name, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels)
    : AbstractImagePixelStream(model, name, imageWidth, imageHeight, nChannels)
{
    tiles_.resize(nTiles());
    for(unsigned int i = 0; i < nTiles(); ++i) {
        unsigned int tileSize = MVMU_DIM;
        if(i == nTiles() - 1 && nChannels%MVMU_DIM > 0) {
            tileSize = nChannels%MVMU_DIM;
        }
        tiles_[i] = new InputImagePixelStreamTile(model, name + "[" + std::to_string(i) + "]", imageWidth, imageHeight, tileSize);
    }
    model->addInputImagePixelStreamImpl(this);
}

OutputVectorImpl::OutputVectorImpl(ModelImpl* model, std::string name, unsigned int length)
    : AbstractVector(model, name, length)
{
    tiles_.resize(nTiles());
    for(unsigned int i = 0; i < nTiles(); ++i) {
        unsigned int tileSize = MVMU_DIM;
        if(i == nTiles() - 1 && length%MVMU_DIM > 0) {
            tileSize = length%MVMU_DIM;
        }
        tiles_[i] = new OutputVectorTile(model, name + "[" + std::to_string(i) + "]", tileSize);
    }
    model->addOutputVectorImpl(this);
}

OutputImagePixelStreamTile::OutputImagePixelStreamTile(ModelImpl* model, std::string name, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels)
    : AbstractImagePixelStream(model, name, imageWidth, imageHeight, nChannels)
{
    stream_.resize(imageHeight);
    for(unsigned int h = 0; h < imageHeight; ++h) {
        stream_[h].resize(imageWidth);
        for(unsigned int w = 0; w < imageWidth; ++w) {
            stream_[h][w] = new OutputVectorTile(model, name + "[" + std::to_string(h) + "][" + std::to_string(w) + "]", nChannels);
        }
    }
}

OutputImagePixelStreamImpl::OutputImagePixelStreamImpl(ModelImpl* model, std::string name, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels)
    : AbstractImagePixelStream(model, name, imageWidth, imageHeight, nChannels)
{
    tiles_.resize(nTiles());
    for(unsigned int i = 0; i < nTiles(); ++i) {
        unsigned int tileSize = MVMU_DIM;
        if(i == nTiles() - 1 && nChannels%MVMU_DIM > 0) {
            tileSize = nChannels%MVMU_DIM;
        }
        tiles_[i] = new OutputImagePixelStreamTile(model, name + "[" + std::to_string(i) + "]", imageWidth, imageHeight, tileSize);
    }
    model->addOutputImagePixelStreamImpl(this);
}

ConstantMatrixImpl::ConstantMatrixImpl(ModelImpl* model, std::string name, unsigned int width, unsigned int height)
    : AbstractMatrix(model, name, width, height)
{
    tiles_.resize(nHeightTiles());
    for(unsigned int h = 0; h < nHeightTiles(); ++h) {
        unsigned int tileHeight = MVMU_DIM;
        if(h == nHeightTiles() - 1 && height%MVMU_DIM > 0) {
            tileHeight = height%MVMU_DIM;
        }
        tiles_[h].resize(nWidthTiles());
        for(unsigned int w = 0; w < nWidthTiles(); ++w) {
            unsigned int tileWidth = MVMU_DIM;
            if(w == nWidthTiles() - 1 && width%MVMU_DIM > 0) {
                tileWidth = width%MVMU_DIM;
            }
            tiles_[h][w] = new ConstantMatrixTile(model, name + "[" + std::to_string(h) + "][" + std::to_string(w) + "]", tileWidth, tileHeight);
        }
    }
    model->addConstantMatrixImpl(this);
}

ConvolutionalConstantMatrixImpl::ConvolutionalConstantMatrixImpl(ModelImpl* model, std::string name, unsigned int kernelWidth, unsigned int kernelHeight, unsigned int nInChannels, unsigned int nOutChannels)
    : AbstractTensor(model, name), kernelWidth_(kernelWidth), kernelHeight_(kernelHeight), nInChannels_(nInChannels), nOutChannels_(nOutChannels)
{
    tiles_.resize(kernelHeight);
    for(unsigned int kh = 0; kh < kernelHeight; ++kh) {
        tiles_[kh].resize(kernelWidth);
        for(unsigned int kw = 0; kw < kernelWidth; ++kw) {
            tiles_[kh][kw].resize(getNOutChannelTiles());
            for(unsigned int h = 0; h < getNOutChannelTiles(); ++h) {
                unsigned int tileHeight = MVMU_DIM;
                if(h == getNOutChannelTiles() - 1 && nOutChannels%MVMU_DIM > 0) {
                    tileHeight = nOutChannels%MVMU_DIM;
                }
                tiles_[kh][kw][h].resize(getNInChannelTiles());
                for(unsigned int w = 0; w < getNInChannelTiles(); ++w) {
                    unsigned int tileWidth = MVMU_DIM;
                    if(w == getNInChannelTiles() - 1 && nInChannels%MVMU_DIM > 0) {
                        tileWidth = nInChannels%MVMU_DIM;
                    }
                    tiles_[kh][kw][h][w] = new ConstantMatrixTile(model, name + "[" + std::to_string(kh) + "][" + std::to_string(kw) + "][" + std::to_string(h) + "][" + std::to_string(w) + "]", tileWidth, tileHeight);
                }
            }
        }
    }
    model->addConvolutionalConstantMatrixImpl(this);
}

TrainingMatrixImpl::TrainingMatrixImpl(ModelImpl* model, std::string name, unsigned int width, unsigned int height)
    : AbstractMatrix(model, name, width, height)
{
    tiles_.resize(nHeightTiles());
    for(unsigned int h = 0; h < nHeightTiles(); ++h) {
        unsigned int tileHeight = MVMU_DIM;
        if(h == nHeightTiles() - 1 && height%MVMU_DIM > 0) {
            tileHeight = height%MVMU_DIM;
        }
        tiles_[h].resize(nWidthTiles());
        for(unsigned int w = 0; w < nWidthTiles(); ++w) {
            unsigned int tileWidth = MVMU_DIM;
            if(w == nWidthTiles() - 1 && width%MVMU_DIM > 0) {
                tileWidth = width%MVMU_DIM;
            }
            tiles_[h][w] = new TrainingMatrixTile(model, name + "[" + std::to_string(h) + "][" + std::to_string(w) + "]", tileWidth, tileHeight);
        }
    }
    model->addTrainingMatrixImpl(this);
}

VectorImpl::VectorImpl(ModelImpl* model, unsigned int length)
    : AbstractVector(model, "", length), tiles_((length - 1)/MVMU_DIM + 1)
{
    model->addVectorImpl(this);
}

ImagePixelStreamTile::ImagePixelStreamTile(ModelImpl* model, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels)
    : AbstractImagePixelStream(model, "", imageWidth, imageHeight, nChannels)
{
    stream_.resize(imageHeight);
    for(unsigned int h = 0; h < imageHeight; ++h) {
        stream_[h].resize(imageWidth);
    }
}

ImagePixelStreamImpl::ImagePixelStreamImpl(ModelImpl* model, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels)
    : AbstractImagePixelStream(model, "", imageWidth, imageHeight, nChannels)
{
    tiles_.resize(nTiles());
    for(unsigned int i = 0; i < nTiles(); ++i) {
        unsigned int tileSize = MVMU_DIM;
        if(i == nTiles() - 1 && nChannels%MVMU_DIM > 0) {
            tileSize = nChannels%MVMU_DIM;
        }
        tiles_[i] = new ImagePixelStreamTile(model, imageWidth, imageHeight, tileSize);
    }
    model->addImagePixelStreamImpl(this);
}

void AbstractVector::checkCompatibility(AbstractVector* v) {
    assert(model_ == v->model_);
    assert(length_ == v->length_);
}

void AbstractImagePixelStream::checkCompatibility(AbstractImagePixelStream* vs) {
    assert(model_ == vs->model_);
    assert(imageWidth_ == vs->imageWidth_);
    assert(imageHeight_ == vs->imageHeight_);
    assert(nChannels_ == vs->nChannels_);
}

void ConstantMatrixImpl::checkCompatibilityForMVM(AbstractVector* v) {
    assert(model_ == v->getModel());
    assert(width_ == v->length());
}

void TrainingMatrixImpl::checkCompatibilityForMVM(AbstractVector* v) {
    assert(model_ == v->getModel());
    assert(width_ == v->length());
}

void TrainingMatrixImpl::checkCompatibilityForMVMTranspose(AbstractVector* v) {
    assert(model_ == v->getModel());
    assert(height_ == v->length());
}

void TrainingMatrixImpl::checkCompatibilityForOuterProductAccumulate(AbstractVector* v1, AbstractVector* v2) {
    assert(model_ == v1->getModel() && model_ == v2->getModel());
    assert(height_ == v1->length());
    assert(width_ == v2->length());
}

void ConvolutionalConstantMatrixImpl::checkCompatibility(AbstractImagePixelStream* vs) {
    assert(model_ == vs->getModel());
    assert(nInChannels_ == vs->nChannels());
}

InputVectorTile* InputVectorImpl::getTile(unsigned int t) {
    assert(tiles_[t] != NULL);
    return tiles_[t];
}

InputImagePixelStreamTile* InputImagePixelStreamImpl::getTile(unsigned int t) {
    assert(tiles_[t] != NULL);
    return tiles_[t];
}

ProducerOperation* VectorImpl::getTile(unsigned int t) {
    assert(tiles_[t] != NULL);
    return tiles_[t];
}

ImagePixelStreamTile* ImagePixelStreamImpl::getTile(unsigned int t) {
    assert(tiles_[t] != NULL);
    return tiles_[t];
}

OutputVectorTile* OutputVectorImpl::getTile(unsigned int t) {
    assert(tiles_[t] != NULL);
    return tiles_[t];
}

OutputImagePixelStreamTile* OutputImagePixelStreamImpl::getTile(unsigned int t) {
    assert(tiles_[t] != NULL);
    return tiles_[t];
}

ConstantMatrixTile* ConstantMatrixImpl::getTile(unsigned int h, unsigned int w) {
    assert(tiles_[h][w] != NULL);
    return tiles_[h][w];
}

ConstantMatrixTile* ConvolutionalConstantMatrixImpl::getTile(unsigned int kh, unsigned int kw, unsigned int h, unsigned int w) {
    assert(tiles_[kh][kw][h][w] != NULL);
    return tiles_[kh][kw][h][w];
}

TrainingMatrixTile* TrainingMatrixImpl::getTile(unsigned int h, unsigned int w) {
    assert(tiles_[h][w] != NULL);
    return tiles_[h][w];
}

void VectorImpl::setTile(unsigned int t, ProducerOperation* producer) {
    assert(tiles_[t] == NULL && "Cannot reassign vector tile");
    tiles_[t] = producer;
}

void ImagePixelStreamTile::add(unsigned int h, unsigned int w, ProducerOperation* vec) {
    assert(vec->length() == nChannels());
    stream_[h][w] = vec;;
}

InputVectorTile* InputImagePixelStreamTile::get(unsigned int h, unsigned int w) {
    return stream_[h][w];
}

ProducerOperation* ImagePixelStreamTile::get(unsigned int h, unsigned int w) {
    return stream_[h][w];
}

OutputVectorTile* OutputImagePixelStreamTile::get(unsigned int h, unsigned int w) {
    return stream_[h][w];
}

InputVectorImpl::~InputVectorImpl() {
    for(InputVectorTile* tile : tiles_) {
        delete tile;
    }
}

InputImagePixelStreamTile::~InputImagePixelStreamTile() {
    for(auto it : stream_) {
        for(InputVectorTile* tile : it) {
            delete tile;
        }
    }
}

InputImagePixelStreamImpl::~InputImagePixelStreamImpl() {
    for(InputImagePixelStreamTile* tile : tiles_) {
        delete tile;
    }
}

ImagePixelStreamImpl::~ImagePixelStreamImpl() {
    for(ImagePixelStreamTile* tile : tiles_) {
        delete tile;
    }
}

OutputVectorImpl::~OutputVectorImpl() {
    for(OutputVectorTile* tile : tiles_) {
        delete tile;
    }
}

OutputImagePixelStreamTile::~OutputImagePixelStreamTile() {
    for(auto it : stream_) {
        for(OutputVectorTile* tile : it) {
            delete tile;
        }
    }
}

OutputImagePixelStreamImpl::~OutputImagePixelStreamImpl() {
    for(OutputImagePixelStreamTile* tile : tiles_) {
        delete tile;
    }
}

ConstantMatrixImpl::~ConstantMatrixImpl() {
    for(auto tileRow : tiles_) {
        for(ConstantMatrixTile* tile : tileRow) {
            delete tile;
        }
    }
}

ConvolutionalConstantMatrixImpl::~ConvolutionalConstantMatrixImpl() {
    for(auto kernelRow : tiles_) {
        for(auto kernelElement : kernelRow) {
            for(auto tileRow : kernelElement) {
                for(ConstantMatrixTile* tile : tileRow) {
                    delete tile;
                }
            }
        }
    }
}

TrainingMatrixImpl::~TrainingMatrixImpl() {
    for(auto tileRow : tiles_) {
        for(TrainingMatrixTile* tile : tileRow) {
            delete tile;
        }
    }
}

std::string AbstractTensor::printNodeName() {
    std::stringstream ss;
    ss << '"' << printTensorType() << "\n" << name_ << '"';
    return ss.str();
}

std::string AbstractImagePixelStream::printNodeName() {
    std::stringstream ss;
    ss << '"' << printTensorType() << "\n" << name_ << '"';
    return ss.str();
}

std::string AbstractTensor::printNodeStyle() {
    return "";
}

std::string InputVectorTile::printNodeStyle() {
    return "[shape=box,style=filled,fillcolor=\"#66CCFF\"]";
}

std::string InputVectorImpl::printNodeStyle() {
    return "[shape=box,style=filled,fillcolor=\"#3399FF\"]";
}

std::string InputImagePixelStreamTile::printNodeStyle() {
    return "[shape=box,style=filled,fillcolor=\"#3399FF\"]";
}

std::string InputImagePixelStreamImpl::printNodeStyle() {
    return "[shape=box,style=filled,fillcolor=\"#3399FF\"]";
}

std::string OutputVectorTile::printNodeStyle() {
    return "[shape=box,style=filled,fillcolor=\"#66CCFF\"]";
}

std::string OutputVectorImpl::printNodeStyle() {
    return "[shape=box,style=filled,fillcolor=\"#3399FF\"]";
}

std::string OutputImagePixelStreamTile::printNodeStyle() {
    return "[shape=box,style=filled,fillcolor=\"#3399FF\"]";
}

std::string OutputImagePixelStreamImpl::printNodeStyle() {
    return "[shape=box,style=filled,fillcolor=\"#3399FF\"]";
}

std::string InputVectorTile::printTensorType() {
    return "InputVectorTile";
}

std::string InputVectorImpl::printTensorType() {
    return "InputVector";
}

std::string InputImagePixelStreamTile::printTensorType() {
    return "InputImagePixelStreamTile";
}

std::string InputImagePixelStreamImpl::printTensorType() {
    return "InputImagePixelStream";
}

std::string VectorImpl::printTensorType() {
    return "Vector";
}

std::string ImagePixelStreamTile::printTensorType() {
    return "ImagePixelStreamTile";
}

std::string ImagePixelStreamImpl::printTensorType() {
    return "ImagePixelStream";
}

std::string OutputVectorTile::printTensorType() {
    return "OutputVectorTile";
}

std::string OutputVectorImpl::printTensorType() {
    return "OutputVector";
}

std::string OutputImagePixelStreamTile::printTensorType() {
    return "OutputImagePixelStreamTile";
}

std::string OutputImagePixelStreamImpl::printTensorType() {
    return "OutputStreamVector";
}

std::string ConstantMatrixTile::printTensorType() {
    return "ConstantMatrixTile";
}

std::string ConstantMatrixImpl::printTensorType() {
    return "ConstantMatrix";
}

std::string ConvolutionalConstantMatrixImpl::printTensorType() {
    return "ConvolutionalConstantMatrix";
}

std::string TrainingMatrixTile::printTensorType() {
    return "TrainingMatrixTile";
}

std::string TrainingMatrixImpl::printTensorType() {
    return "TrainingMatrix";
}

void InputVectorImpl::printNodeAndEdges(std::ostream& fout) {
    fout << printNodeName() << " " << printNodeStyle() << ";" << std::endl;
    for(InputVectorTile* tile : tiles_) {
        fout << tile->printNodeName() << " " << tile->printNodeStyle() << ";" << std::endl;
        fout << printNodeName() << " -> " << tile->printNodeName() << " [style=dotted];" << std::endl;
        // NOTE: edges from tiles to their users are printed by the users in InputOperation::printNodeAndEdges
    }
}

void InputImagePixelStreamImpl::printNodeAndEdges(std::ostream& fout) {
    fout << printNodeName() << " " << printNodeStyle() << ";" << std::endl;
    for(InputImagePixelStreamTile* tile : tiles_) {
        fout << tile->printNodeName() << " " << tile->printNodeStyle() << ";" << std::endl;
        fout << printNodeName() << " -> " << tile->printNodeName() << " [style=dotted];" << std::endl;
        for(unsigned int h = 0; h < tile->imageHeight(); ++h) {
            for(unsigned int w = 0; w < tile->imageWidth(); ++w) {
                InputVectorTile* streamElement = tile->get(h, w);
                fout << streamElement->printNodeName() << " " << streamElement->printNodeStyle() << ";" << std::endl;
                fout << tile->printNodeName() << " -> " << streamElement->printNodeName() << " [style=dotted];" << std::endl;
                // NOTE: edges from stream elements to their users are printed by the users in InputOperation::printNodeAndEdges
            }
        }
    }
}

void OutputVectorImpl::printNodeAndEdges(std::ostream& fout) {
    fout << printNodeName() << " " << printNodeStyle() << ";" << std::endl;
    for(OutputVectorTile* tile : tiles_) {
        fout << tile->printNodeName() << " " << tile->printNodeStyle() << ";" << std::endl;
        fout << tile->printNodeName() << " -> " << printNodeName() << " [style=dotted];" << std::endl;
        // NOTE: edges to tiles from their sources are printed by the sources in OutputOperation::printNodeAndEdges
    }
}

void OutputImagePixelStreamImpl::printNodeAndEdges(std::ostream& fout) {
    fout << printNodeName() << " " << printNodeStyle() << ";" << std::endl;
    for(OutputImagePixelStreamTile* tile : tiles_) {
        fout << tile->printNodeName() << " " << tile->printNodeStyle() << ";" << std::endl;
        fout << tile->printNodeName() << " -> " << printNodeName() << " [style=dotted];" << std::endl;
        for(unsigned int h = 0; h < tile->imageHeight(); ++h) {
            for(unsigned int w = 0; w < tile->imageWidth(); ++w) {
                OutputVectorTile* streamElement = tile->get(h, w);
                fout << streamElement->printNodeName() << " " << streamElement->printNodeStyle() << ";" << std::endl;
                fout << streamElement->printNodeName() << " -> " << tile->printNodeName() << " [style=dotted];" << std::endl;
                // NOTE: edges to stream elements from their sources are printed by the users in OutputOperation::printNodeAndEdges
            }
        }
    }
}

