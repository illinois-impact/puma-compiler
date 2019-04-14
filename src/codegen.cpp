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

#include "puma.h"

#include "coalescer.h"
#include "codegen.h"
#include "linearizer.h"
#include "memalloc.h"
#include "model.h"
#include "operations.h"
#include "placer.h"
#include "regalloc.h"

CodeGenerator::CodeGenerator(ModelImpl* model, Placer* placer, MemoryAllocator* memoryAllocator, Coalescer* coalescer, Linearizer* linearizer, RegisterAllocator* registerAllocator)
    : model_(model), placer_(placer), memoryAllocator_(memoryAllocator), coalescer_(coalescer), linearizer_(linearizer), registerAllocator_(registerAllocator)
{
    codegen();
}

void CodeGenerator::codegen() {

    // TODO: Define ABI for laying out the binary

    for(unsigned int pTile = 0; pTile < placer_->getNPTiles(); ++pTile) {

        // Generate code for the tile
        std::stringstream fileName;
        fileName << model_->getName() << "-tile" << pTile << ".puma";
        std::ofstream tileCode;
        tileCode.open(fileName.str());
        std::list<TileOperation*>& tileOperationList = linearizer_->getTileOperationList(pTile);
        for(TileOperation* tileOp : tileOperationList) {
            if(SendOperation* send = dynamic_cast<SendOperation*>(tileOp)) {
                tileCode << codegen(send);
            } else if(ReceiveOperation* recv = dynamic_cast<ReceiveOperation*>(tileOp)) {
                tileCode << codegen(recv);
            } else if(WriteInputOperation* write = dynamic_cast<WriteInputOperation*>(tileOp)) {
                tileCode << codegen(write);
            } else if(ReadOutputOperation* read = dynamic_cast<ReadOutputOperation*>(tileOp)) {
                tileCode << codegen(read);
            } else {
                assert(0 && "Unsupported operation for code generation!");
            }
        }
        tileCode << "halt()" << std::endl;
        tileCode.close();

        // Generate code for each core in the tile
        for(unsigned int pCore = 0; pCore < N_CORES_PER_TILE; ++pCore) {
            std::stringstream fileName;
            fileName << model_->getName() << "-tile" << pTile << "-core" << pCore << ".puma";
            std::ofstream coreCode;
            coreCode.open(fileName.str());
            std::list<CoreOperation*>& coreOperationList = linearizer_->getCoreOperationList(pTile, pCore);
            for(CoreOperation* coreOp : coreOperationList) {
                if(MVMOperation* mvm = dynamic_cast<MVMOperation*>(coreOp)) {
                    coreCode << codegen(mvm);
                } else if(TrainingMatrixOperation* trainOp = dynamic_cast<TrainingMatrixOperation*>(coreOp)) {
                    coreCode << codegen(trainOp);
                } else if(ALUVectorOperation* aluOp = dynamic_cast<ALUVectorOperation*>(coreOp)) {
                    coreCode << codegen(aluOp);
                } else if(SetImmediateOperation* seti = dynamic_cast<SetImmediateOperation*>(coreOp)) {
                    coreCode << codegen(seti);
                } else if(CopyOperation* copy = dynamic_cast<CopyOperation*>(coreOp)) {
                    coreCode << codegen(copy);
                } else if(LoadOperation* load = dynamic_cast<LoadOperation*>(coreOp)) {
                    coreCode << codegen(load);
                } else if(StoreOperation* store = dynamic_cast<StoreOperation*>(coreOp)) {
                    coreCode << codegen(store);
                } else {
                    assert(0 && "Unsupported operation for code generation!");
                }
            }
            coreCode << "hlt()" << std::endl;
            coreCode.close();
        }

    }

}

std::string CodeGenerator::codegen(CoalescedMVMSet* coalescedMVMSet) {
    std::stringstream ss;
    ss << "mvm(['";
    for(unsigned int i = 0; i < N_CONSTANT_MVMUS_PER_CORE; ++i) {
        if(coalescedMVMSet->usesPMVMU(i)) {
            ss << 1;
        } else {
            ss << 0;
        }
    }
    ss << "'])\n";
    return ss.str();
}

std::string CodeGenerator::codegen(CoalescedTrainingOperationSet* coalescedTrainingOperationSet) {
    std::stringstream ss;
    ss << "train([";
    for(unsigned int pMVMU = 0; pMVMU < N_TRAINING_MVMUS_PER_CORE; ++pMVMU) {
        ss << "'";
        for(unsigned int t = 0; t < N_TRAINING_OPERATIONS; ++t) {
            TrainingMatrixOperation::OpType opType = (TrainingMatrixOperation::OpType)t;
            if(coalescedTrainingOperationSet->usesPMVMUForOp(pMVMU, opType)) {
                ss << 1;
            } else {
                ss << 0;
            }
        }
        ss << "'";
    }
    ss << "])\n";
    return ss.str();
}

std::string CodeGenerator::codegen(MVMOperation* mvm) {
    CoalescedMVMSet* coalescedMVMSet = mvm->getCoalescedSet();
    if(coalescedMVMSet != NULL) {
        if(coalescedMVMSet->isSetLeader(mvm)) { // Only one MVM in a coalesced set does code generation on behalf of the others
            return codegen(coalescedMVMSet);
        } else {
            return "";
        }
    } else {
        std::stringstream ss;
        ss << "mvm(['";
        for(unsigned int i = 0; i < N_CONSTANT_MVMUS_PER_CORE; ++i) {
            if(i == placer_->getPMVMU(mvm)) {
                ss << 1;
            } else {
                ss << 0;
            }
        }
        ss << "'])\n";
        return ss.str();
    }
}

std::string CodeGenerator::codegen(TrainingMatrixOperation* trainOp) {
    CoalescedTrainingOperationSet* coalescedTrainingOperationSet = trainOp->getCoalescedSet();
    if(coalescedTrainingOperationSet != NULL) {
        if(coalescedTrainingOperationSet->isSetLeader(trainOp)) { // Only one training operation in a coalesced set does code generation on behalf of the others
            return codegen(coalescedTrainingOperationSet);
        } else {
            return "";
        }
    } else {
        std::stringstream ss;
        ss << "train([";
        for(unsigned int pMVMU = 0; pMVMU < N_TRAINING_MVMUS_PER_CORE; ++pMVMU) {
            ss << "'";
            for(unsigned int t = 0; t < N_TRAINING_OPERATIONS; ++t) {
                TrainingMatrixOperation::OpType opType = (TrainingMatrixOperation::OpType)t;
                if(pMVMU == placer_->getPMVMU(trainOp) && opType == trainOp->getOpType()) {
                    ss << 1;
                } else {
                    ss << 0;
                }
            }
        }
        ss << "])\n";
        return ss.str();
    }
}

std::string CodeGenerator::codegen(ALUVectorOperation* aluOp) {
    std::stringstream ss;
    ss << "alu";
    switch(aluOp->getOpCode()) {
        case ALUVectorOperation::MULI:
            ss << "i";
    }
    ss << "('";
    switch(aluOp->getOpCode()) {
        case ALUVectorOperation::ADD: ss << "add"; break;
        case ALUVectorOperation::SUB: ss << "sub"; break;
        case ALUVectorOperation::MUL:
        case ALUVectorOperation::MULI: ss << "mul"; break;
        case ALUVectorOperation::DIV: ss << "div"; break;
        case ALUVectorOperation::AND: ss << "and"; break;
        case ALUVectorOperation::OR: ss << "or"; break;
        case ALUVectorOperation::NOT: ss << "not"; break;
        case ALUVectorOperation::EQ: ss << "eq"; break;
        case ALUVectorOperation::NEQ: ss << "neq"; break;
        case ALUVectorOperation::LT: ss << "lt"; break;
        case ALUVectorOperation::LEQ: ss << "leq"; break;
        case ALUVectorOperation::GT: ss << "gt"; break;
        case ALUVectorOperation::GEQ: ss << "geq"; break;
        case ALUVectorOperation::MIN: ss << "min"; break;
        case ALUVectorOperation::MAX: ss << "max"; break;
        case ALUVectorOperation::MSE: ss << "mse"; break;
        case ALUVectorOperation::SIG: ss << "sig"; break;
        case ALUVectorOperation::TANH: ss << "tanh"; break;
        case ALUVectorOperation::EXP: ss << "exp"; break;
        case ALUVectorOperation::LOG: ss << "log"; break;
        case ALUVectorOperation::RELU: ss << "relu"; break;
        case ALUVectorOperation::RELUD: ss << "relud"; break;
        case ALUVectorOperation::LOG_SOFTMAX: ss << "log_softmax"; break;
        case ALUVectorOperation::LOG_SOFTMAXD: ss << "log_softmaxd"; break;
        case ALUVectorOperation::RNDCMP: ss << "rndcmp"; break;
    }
    ss << "', "
       << "d1=" << registerAllocator_->getRegister(aluOp) << ", "
       << "r1=" << registerAllocator_->getRegister(aluOp->getOperand(0)) << ", ";
    if(aluOp->numOperands() > 1) {
        ss << "r2=" << registerAllocator_->getRegister(aluOp->getOperand(1)) << ", ";
    }
    if(aluOp->isImmediate()) {
        ss << "imm=" << aluOp->getImmediate() << ", ";
    }
    ss << "vec=" << aluOp->length()
       << ")\n";
    return ss.str();
}

std::string CodeGenerator::codegen(SetImmediateOperation* seti) {
    std::stringstream ss;
    ss << "set("
       << "d1=" << registerAllocator_->getRegister(seti) << ", "
       << "imm=" << seti->getImmediate() << ", "
       << "vec=" << seti->length()
       << ")\n";
    return ss.str();
}

std::string CodeGenerator::codegen(CopyOperation* copy) {
    std::stringstream ss;
    ss << "copy("
       << "d1=" << registerAllocator_->getRegister(copy) << ", "
       << "r1=" << registerAllocator_->getRegister(copy->getOperand(0)) << ", "
       << "vec=" << copy->length() << ", "
       << "src_type=" << 1
       << ")\n";
    return ss.str();
}

std::string CodeGenerator::codegen(LoadOperation* load) {
    std::stringstream ss;
    unsigned int loadWidth;
    for(loadWidth = MAX_LOAD_STORE_WIDTH; !(load->length()%loadWidth == 0); --loadWidth);
    ss << "load("
       << "d1=" << registerAllocator_->getRegister(load) << ", "
       << "r1=" << registerAllocator_->getRegister(load->getOperand(0)) << ", "
       << "load_width=" << loadWidth << ", "
       << "vec=" << load->length()/loadWidth
       << ")\n";
    return ss.str();
}

std::string CodeGenerator::codegen(StoreOperation* store) {
    std::stringstream ss;
    unsigned int storeWidth;
    for(storeWidth = MAX_LOAD_STORE_WIDTH; !(store->length()%storeWidth == 0); --storeWidth);
    ss << "store(d1=" << registerAllocator_->getRegister(store->getOperand(1)) << ", "
       << "r1=" << registerAllocator_->getRegister(store->getOperand(0)) << ", "
       << "counter=" << store->numUsers() << ", "
       << "store_width=" << storeWidth << ", "
       << "vec=" << store->length()/storeWidth
       << ")\n";
    return ss.str();
}

std::string CodeGenerator::codegen(SendOperation* send) {
    std::stringstream ss;
    unsigned int sendWidth;
    for(sendWidth = MAX_SEND_RECV_WIDTH; !(send->length()%sendWidth == 0); --sendWidth);
    ss << "send("
       << "mem_addr=" << memoryAllocator_->getTileMemoryAddress(send->getSrc(0)) << ", "
       << "vtile_id=" << placer_->getPTile(send) << ", " // FIXME: Assign sender IDs
       << "send_width=" << sendWidth << ", "
       << "target_addr=" << placer_->getPTile(send->getDst()) << ", "
       << "vec=" << send->length()/sendWidth
       << ")\n";
    return ss.str();
}

std::string CodeGenerator::codegen(ReceiveOperation* recv) {
    std::stringstream ss;
    unsigned int recvWidth;
    for(recvWidth = MAX_SEND_RECV_WIDTH; !(recv->length()%recvWidth == 0); --recvWidth);
    ss << "receive(mem_addr=" << memoryAllocator_->getTileMemoryAddress(recv) << ", "
       << "vtile_id=" << placer_->getPTile(recv->getSrc()) << ", " // FIXME: Assign sender IDs
       << "receive_width=" << recvWidth << ", "
       << "counter=" << recv->numUsers() << ", "
       << "vec=" << recv->length()/recvWidth
       << ")\n";
    return ss.str();
}

std::string CodeGenerator::codegen(WriteInputOperation* write) {
    return "";
}

std::string CodeGenerator::codegen(ReadOutputOperation* read) {
    return "";
}

