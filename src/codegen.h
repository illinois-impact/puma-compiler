/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "common.h"

class CodeGenerator {

    private:

        ModelImpl* model_;
        Placer* placer_;
        MemoryAllocator* memoryAllocator_;
        Coalescer* coalescer_;
        Linearizer* linearizer_;
        RegisterAllocator* registerAllocator_;

        void codegen();
        std::string codegen(CoalescedMVMSet* coalescedMVMSet);
        std::string codegen(CoalescedTrainingOperationSet* coalescedTrainingOperationSet);
        std::string codegen(MVMOperation* mvm);
        std::string codegen(TrainingMatrixOperation* trainOp);
        std::string codegen(ALUVectorOperation* aluOp);
        std::string codegen(SetImmediateOperation* seti);
        std::string codegen(CopyOperation* copy);
        std::string codegen(LoadOperation* load);
        std::string codegen(StoreOperation* store);
        std::string codegen(SendOperation* send);
        std::string codegen(ReceiveOperation* recv);
        std::string codegen(WriteInputOperation* write);
        std::string codegen(ReadOutputOperation* read);

    public:

        CodeGenerator(ModelImpl* model, Placer* placer, MemoryAllocator* memoryAllocator, Coalescer* coalescer, Linearizer* linearizer, RegisterAllocator* registerAllocator);

};

