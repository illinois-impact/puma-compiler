/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include "puma.h"

#include "coalescer.h"
#include "codegen.h"
#include "instance.h"
#include "linearizer.h"
#include "memalloc.h"
#include "model.h"
#include "operations.h"
#include "partitioner.h"
#include "placer.h"
#include "regalloc.h"
#include "tensors.h"

Model Model::create(std::string name) {
    Model model;
    model.impl_ = new ModelImpl(name);
    return model;
}

void Model::destroy() {
    delete impl_;
}

ModelImpl* Model::unwrap() {
    return impl_;
}

void Model::compile(CompilerOptions options) {
    impl_->compile(options);
}

ModelImpl::ModelImpl(std::string name)
    : name_(name), modelType_(UNSPECIALIZED), partitioner_(NULL), placer_(NULL), memoryAllocator_(NULL), coalescer_(NULL), linearizer_(NULL), registerAllocator_(NULL), codeGenerator_(NULL)
{ }

ModelImpl::~ModelImpl() {
    if(partitioner_ != NULL) {
        delete partitioner_;
    }
    if(placer_ != NULL) {
        delete placer_;
    }
    if(memoryAllocator_ != NULL) {
        delete memoryAllocator_;
    }
    if(coalescer_ != NULL) {
        delete coalescer_;
    }
    if(linearizer_ != NULL) {
        delete linearizer_;
    }
    if(registerAllocator_ != NULL) {
        delete registerAllocator_;
    }
    if(codeGenerator_ != NULL) {
        delete codeGenerator_;
    }
    for(InputVectorImpl* vec : inputVectors_) {
        delete vec;
    }
    for(InputImagePixelStreamImpl* stream : inputImagePixelStreams_) {
        delete stream;
    }
    for(VectorImpl* vec : vectors_) {
        delete vec;
    }
    for(ImagePixelStreamImpl* stream : imagePixelStreams_) {
        delete stream;
    }
    for(OutputVectorImpl* vec : outputVectors_) {
        delete vec;
    }
    for(OutputImagePixelStreamImpl* stream : outputImagePixelStreams_) {
        delete stream;
    }
    for(ConstantMatrixImpl* matrix : constantMatrices_) {
        delete matrix;
    }
    for(ConvolutionalConstantMatrixImpl* matrix : convolutionMatrices_) {
        delete matrix;
    }
    for(TrainingMatrixImpl* matrix : trainingMatrices_) {
        delete matrix;
    }
    for(Operation* op : operations_) {
        delete op;
    }
    for(auto coalesceableMVMSet : coalesceableMVMSets_) {
        delete coalesceableMVMSet;
    }
    for(auto instance : instances_) {
        delete instance;
    }
}

void ModelImpl::addInputVectorImpl(InputVectorImpl* vec) {
    inputVectors_.push_back(vec);
}

void ModelImpl::addInputImagePixelStreamImpl(InputImagePixelStreamImpl* stream) {
    inputImagePixelStreams_.push_back(stream);
}

void ModelImpl::addVectorImpl(VectorImpl* vec) {
    vectors_.push_back(vec);
}

void ModelImpl::addImagePixelStreamImpl(ImagePixelStreamImpl* stream) {
    imagePixelStreams_.push_back(stream);
}

void ModelImpl::addOutputVectorImpl(OutputVectorImpl* vec) {
    outputVectors_.push_back(vec);
}

void ModelImpl::addOutputImagePixelStreamImpl(OutputImagePixelStreamImpl* stream) {
    outputImagePixelStreams_.push_back(stream);
}

void ModelImpl::addConstantMatrixImpl(ConstantMatrixImpl* mat) {
    if(modelType_ == UNSPECIALIZED) {
        modelType_ = INFERENCE;
    } else {
        assert(modelType_ == INFERENCE && "Cannot mix inference and training matrices in the same model");
    }
    constantMatrices_.push_back(mat);
}

void ModelImpl::addConvolutionalConstantMatrixImpl(ConvolutionalConstantMatrixImpl* mat) {
    if(modelType_ == UNSPECIALIZED) {
        modelType_ = INFERENCE;
    } else {
        assert(modelType_ == INFERENCE && "Cannot mix inference and training matrices in the same model");
    }
    convolutionMatrices_.push_back(mat);
}

void ModelImpl::addTrainingMatrixImpl(TrainingMatrixImpl* mat) {
    if(modelType_ == UNSPECIALIZED) {
        modelType_ = TRAINING;
    } else {
        assert(modelType_ == TRAINING && "Cannot mix inference and training matrices in the same model");
    }
    trainingMatrices_.push_back(mat);
}

void ModelImpl::addOperation(Operation* op) {
    operations_.insert(op);
}

void ModelImpl::addCoalesceableMVMSet(std::set<MVMOperation*>* coalesceableMVMSet) {
    coalesceableMVMSets_.push_back(coalesceableMVMSet);
}

void ModelImpl::unlink(Operation* op) {
    operations_.erase(op);
    delete op;
}

void ModelImpl::printGraph(std::string fileName) {
    std::ofstream fout;
    fout.open(fileName);
    fout << "digraph model {" << std::endl;
    for(InputVectorImpl* vec : inputVectors_) {
        vec->printNodeAndEdges(fout);
    }
    for(InputImagePixelStreamImpl* stream : inputImagePixelStreams_) {
        stream->printNodeAndEdges(fout);
    }
    for(OutputVectorImpl* vec : outputVectors_) {
        vec->printNodeAndEdges(fout);
    }
    for(OutputImagePixelStreamImpl* stream : outputImagePixelStreams_) {
        stream->printNodeAndEdges(fout);
    }
    for(Operation* op : operations_) {
        op->printNodeAndEdges(fout);
    }
    fout << "}" << std::endl;
    fout.close();
}

std::string ModelImpl::printAssignment(Operation* op) {
    std::stringstream ss;
    if(partitioner_ != NULL) {
        ss << partitioner_->printAssignment(op);
    }
    if(placer_ != NULL) {
        ss << placer_->printAssignment(op);
    }
    if(memoryAllocator_ != NULL) {
        ss << memoryAllocator_->printAssignment(op);
    }
    if(registerAllocator_ != NULL) {
        ss << registerAllocator_->printAssignment(op);
    }
    return ss.str();
}

void ModelImpl::compile(CompilerOptions& options) {

    if(options.printDebugInfo_) {
        printGraph(name_ + "-graph0.dot");
    }

    // Model partitioning
    std::cout << "Partitioning graph... " << std::flush;
    partitioner_ = new Partitioner(this, options.gp_);
    std::cout << "done." << std::endl;
    if(options.printDebugInfo_) {
        printGraph(name_ + "-graph1-partitioned.dot");
    }

    // Physical layout
    std::cout << "Physical layout... " << std::flush;
    placer_ = new Placer(this, partitioner_);
    std::cout << "done." << std::endl;
    if(options.printDebugInfo_) {
        printGraph(name_ + "-graph2-virtual-to-physical.dot");
    }

    // Memory allocation
    std::cout << "Memory allocation... " << std::flush;
    memoryAllocator_ = new MemoryAllocator(this, partitioner_);
    std::cout << "done." << std::endl;
    if(options.printDebugInfo_) {
        printGraph(name_ + "-graph3-memory-allocation.dot");
    }

    // Coalescing
    if(options.coalesceMVMOperations_) {
        std::cout << "MVM coalescing... " << std::flush;
        coalescer_ = new Coalescer(this, placer_, coalesceableMVMSets_);
        std::cout << "done." << std::endl;
    }

    // Linearization
    std::cout << "Linearizing graph... " << std::flush;
    linearizer_ = new Linearizer(this, partitioner_, placer_);
    std::cout << "done." << std::endl;
    if(options.printDebugInfo_) {
        printGraph(name_ + "-graph4-linearization.dot");
    }

    // Register allocation
    std::cout << "Register allocation... " << std::flush;
    registerAllocator_ = new RegisterAllocator(this, partitioner_, placer_, memoryAllocator_, linearizer_);
    std::cout << "done." << std::endl;
    if(options.printDebugInfo_) {
        printGraph(name_ + "-graph5-register-allocation.dot");
    }

    // Code generation
    std::cout << "Code generation... " << std::flush;
    codeGenerator_ = new CodeGenerator(this, placer_, memoryAllocator_, coalescer_, linearizer_, registerAllocator_);
    std::cout << "done." << std::endl;

    // Report
    std::ofstream report(name_ + "-report.out");
    partitioner_->printReport(report);
    registerAllocator_->printReport(report);
    report.close();

}

ModelInstanceImpl* ModelImpl::createInstance() {
    ModelInstanceImpl* instance = new ModelInstanceImpl(this, placer_);
    instances_.insert(instance);
    return instance;
}

