/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common.h"

class ModelImpl {

    public:

        enum ModelType { UNSPECIALIZED, INFERENCE, TRAINING };

    private:

        std::string name_;
        ModelType modelType_;
        std::vector<InputVectorImpl*> inputVectors_;
        std::vector<InputImagePixelStreamImpl*> inputImagePixelStreams_;
        std::vector<VectorImpl*> vectors_;
        std::vector<ImagePixelStreamImpl*> imagePixelStreams_;
        std::vector<OutputVectorImpl*> outputVectors_;
        std::vector<OutputImagePixelStreamImpl*> outputImagePixelStreams_;
        std::vector<ConstantMatrixImpl*> constantMatrices_;
        std::vector<ConvolutionalConstantMatrixImpl*> convolutionMatrices_;
        std::vector<TrainingMatrixImpl*> trainingMatrices_;
        std::set<Operation*> operations_;
        std::vector<std::set<MVMOperation*>*> coalesceableMVMSets_;

        Partitioner* partitioner_;
        Placer* placer_;
        MemoryAllocator* memoryAllocator_;
        Coalescer* coalescer_;
        Linearizer* linearizer_;
        RegisterAllocator* registerAllocator_;
        CodeGenerator* codeGenerator_;

        std::set<ModelInstanceImpl*> instances_;

        // Debug information
        void printGraph(std::string fileName);

    public:

        ModelImpl(std::string name);
        ~ModelImpl();

        void addInputVectorImpl(InputVectorImpl* vec);
        void addInputImagePixelStreamImpl(InputImagePixelStreamImpl* stream);
        void addVectorImpl(VectorImpl* vec);
        void addImagePixelStreamImpl(ImagePixelStreamImpl* stream);
        void addOutputVectorImpl(OutputVectorImpl* vec);
        void addOutputImagePixelStreamImpl(OutputImagePixelStreamImpl* stream);
        void addConstantMatrixImpl(ConstantMatrixImpl* mat);
        void addConvolutionalConstantMatrixImpl(ConvolutionalConstantMatrixImpl* mat);
        void addTrainingMatrixImpl(TrainingMatrixImpl* mat);
        void addOperation(Operation* op);
        void addCoalesceableMVMSet(std::set<MVMOperation*>* coalesceableMVMSet);

        void unlink(Operation* op);

        void compile(CompilerOptions& options);

        ModelInstanceImpl* createInstance();

        std::string getName() { return name_; }
        ModelType getModelType() { return modelType_; }

        // Iterators
        std::vector<ConstantMatrixImpl*>::iterator const_mat_begin() { return constantMatrices_.begin(); }
        std::vector<ConstantMatrixImpl*>::iterator const_mat_end() { return constantMatrices_.end(); }
        std::vector<ConvolutionalConstantMatrixImpl*>::iterator conv_mat_begin() { return convolutionMatrices_.begin(); }
        std::vector<ConvolutionalConstantMatrixImpl*>::iterator conv_mat_end() { return convolutionMatrices_.end(); }
        std::vector<TrainingMatrixImpl*>::iterator train_mat_begin() { return trainingMatrices_.begin(); }
        std::vector<TrainingMatrixImpl*>::iterator train_mat_end() { return trainingMatrices_.end(); }
        std::set<Operation*>::iterator op_begin() { return operations_.begin(); }
        std::set<Operation*>::iterator op_end() { return operations_.end(); }

        // Debug information
        std::string printAssignment(Operation* op);

};

