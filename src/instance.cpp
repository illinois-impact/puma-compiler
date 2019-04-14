/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>
#include <fstream>
#include <sstream>

#include "instance.h"
#include "model.h"
#include "placer.h"
#include "tensors.h"

ModelInstance ModelInstance::create(Model model) {
    ModelInstance instance;
    instance.impl_ = model.unwrap()->createInstance();
    return instance;
}

void ModelInstance::bind(std::string tensorName, float* data) {
    impl_->bind(tensorName, data);
}

void ModelInstance::generateData() {
    impl_->generateData();
}

ModelInstanceImpl* ModelInstance::unwrap() {
    return impl_;
}

ModelInstanceImpl::ModelInstanceImpl(ModelImpl* model, Placer* placer)
    : model_(model), placer_(placer)
{ }

void ModelInstanceImpl::bind(std::string tensorName, float* data) {
    tensorData_[tensorName] = data;
}

void ModelInstanceImpl::generateData() {

    // TODO: Define ABI for laying out the data

    std::cout << "Generating data files... " << std::flush;

    for(auto m = model_->const_mat_begin(); m != model_->const_mat_end(); ++m) {
        ConstantMatrixImpl* mat = *m;
        std::string matName = mat->name();
        assert(tensorData_.count(matName) && "No data provided for matrix");
        float* matData = tensorData_[matName];
        for(unsigned int h = 0; h < mat->nHeightTiles(); ++h) {
            for(unsigned int w = 0; w < mat->nWidthTiles(); ++w) {
                ConstantMatrixTile* matTile = mat->getTile(h, w);
                unsigned int pTile = placer_->getPTile(matTile);
                unsigned int pCore = placer_->getPCore(matTile);
                unsigned int pMVMU = placer_->getPMVMU(matTile);
                std::stringstream fileName;
                fileName << model_->getName() << "-tile" << pTile << "-core" << pCore << "-mvmu" << pMVMU << ".weights";
                std::ofstream mvmuData;
                mvmuData.open(fileName.str());
                for(unsigned int row = 0; row < MVMU_DIM; ++row) {
                    for(unsigned int col = 0; col < MVMU_DIM; ++col) {
                        if(row < matTile->height() && col < matTile->width()) {
                            mvmuData << matData[(h*MVMU_DIM + row)*mat->width() + w*MVMU_DIM + col] << " ";
                        } else {
                            mvmuData << "0.0 ";
                        }
                    }
                }
                mvmuData.close();
            }
        }
    }
    for(auto m = model_->conv_mat_begin(); m != model_->conv_mat_end(); ++m) {
        ConvolutionalConstantMatrixImpl* mat = *m;
        std::string matName = mat->name();
        assert(tensorData_.count(matName) && "No data provided for matrix");
        float* matData = tensorData_[matName];
        for(unsigned int kh = 0; kh < mat->getKernelHeight(); ++kh) {
            for(unsigned int kw = 0; kw < mat->getKernelWidth(); ++kw) {
                for(unsigned int h = 0; h < mat->getNOutChannelTiles(); ++h) {
                    for(unsigned int w = 0; w < mat->getNInChannelTiles(); ++w) {
                        ConstantMatrixTile* matTile = mat->getTile(kh, kw, h, w);
                        unsigned int pTile = placer_->getPTile(matTile);
                        unsigned int pCore = placer_->getPCore(matTile);
                        unsigned int pMVMU = placer_->getPMVMU(matTile);
                        std::stringstream fileName;
                        fileName << model_->getName() << "-tile" << pTile << "-core" << pCore << "-mvmu" << pMVMU << ".weights";
                        std::ofstream mvmuData;
                        mvmuData.open(fileName.str());
                        for(unsigned int row = 0; row < MVMU_DIM; ++row) {
                            for(unsigned int col = 0; col < MVMU_DIM; ++col) {
                                if(row < matTile->height() && col < matTile->width()) {
                                    mvmuData << matData[((kh*mat->getKernelWidth() + kw)*mat->getNOutChannels() + h*MVMU_DIM + row)*mat->getNInChannels() + w*MVMU_DIM + col] << " ";
                                } else {
                                    mvmuData << "0.0 ";
                                }
                            }
                        }
                        mvmuData.close();
                    }
                }
            }
        }
    }

    std::cout << "done." << std::endl;

}

