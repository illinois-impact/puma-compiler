/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>

#include "puma.h"

#include "coalescer.h"
#include "model.h"
#include "operations.h"
#include "placer.h"

Coalescer::Coalescer(ModelImpl* model, Placer* placer, std::vector<std::set<MVMOperation*>*>& coalesceableMVMSets)
    : model_(model), placer_(placer), coalesceableMVMSets_(coalesceableMVMSets)
{
    if(model_->getModelType() == ModelImpl::INFERENCE) {
        coalesceMVMOperations();
    } else {
        coalesceTrainingOperations();
    }
}

Coalescer::~Coalescer() {
    for(auto coreCoalescedSets : coalescedMVMSets_) {
        for(auto coalescedSet : coreCoalescedSets) {
            delete coalescedSet;
        }
    }
    for(auto coreCoalescedSets : coalescedTrainingOperationSets_) {
        for(auto coalescedSet : coreCoalescedSets) {
            delete coalescedSet;
        }
    }
}

void Coalescer::coalesceMVMOperations() {

    coalescedMVMSets_.resize(placer_->getNPCores());

    // Coalesce MVM operations that are known to be coalesceable
    for(auto coalesceableMVMSet : coalesceableMVMSets_) {
        // Extract coalesced set for each relevant core
        std::map<unsigned int, std::map<unsigned int, CoalescedMVMSet*>> localCoalescedMVMSets;
        for(auto mvm : *coalesceableMVMSet) {
            unsigned int pMVMU = placer_->getPMVMU(mvm);
            unsigned int pCore = placer_->getPCore(mvm);
            unsigned int pTile = placer_->getPTile(mvm);
            if(!localCoalescedMVMSets[pTile].count(pCore)) {
                localCoalescedMVMSets[pTile][pCore] = new CoalescedMVMSet();
            }
            localCoalescedMVMSets[pTile][pCore]->add(mvm, pMVMU);
        }
        // Add extracted sets to full list
        for(auto it1 : localCoalescedMVMSets) {
            unsigned int pTile = it1.first;
            for(auto it2 : it1.second) {
                unsigned int pCore = it2.first;
                CoalescedMVMSet* coalescedSet = it2.second;
                if(coalescedSet->isComplete()) {
                    coalescedMVMSets_[pTile*N_CORES_PER_TILE + pCore].push_back(coalescedSet);
                } else {
                    // Only keep complete sets so that MVMs in different incomplete sets can still be coalesced together later
                    coalescedSet->removeAll();
                    delete coalescedSet;
                }
            }
        }
    }

    // Analyze initial dependences between remaining MVM operations
    std::map<Operation*, std::set<MVMOperation*>> mvmPredecessors;
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if(dynamic_cast<ReadOutputOperation*>(op)) {
            findMVMPredecessors(op, mvmPredecessors);
        }
    }

    // Extract useful information
    std::map<MVMOperation*, std::set<MVMOperation*>> mvmPredecessorsOfMVMs;
    std::map<MVMOperation*, std::set<MVMOperation*>> mvmSuccessorsOfMVMs;
    for(auto it : mvmPredecessors) {
        Operation* op = it.first;
        if(MVMOperation* mvm = dynamic_cast<MVMOperation*>(op)) {
            for(MVMOperation* predMVM : it.second) {
                mvmPredecessorsOfMVMs[mvm].insert(predMVM);
                mvmSuccessorsOfMVMs[predMVM].insert(mvm);
            }
        }
    }

    // Coalesce MVMs (in linearization order)
    std::set<Operation*> isVisited;
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if(dynamic_cast<ReadOutputOperation*>(op)) {
            coalesceMVMPredecessors(op, isVisited, mvmPredecessorsOfMVMs, mvmSuccessorsOfMVMs);
        }
    }

}

void Coalescer::findMVMPredecessors(Operation* op, std::map<Operation*, std::set<MVMOperation*>>& mvmPredecessors) {
    if(!mvmPredecessors.count(op)) {
        // Visit nodes in reverse postorder (find MVM predecessors of all predecessors of the operation to determine predecessors of self)
        mvmPredecessors[op]; // Initialize as empty
        if(MVMOperation* mvm = dynamic_cast<MVMOperation*>(op)) {
            CoalescedMVMSet* coalescedSet = mvm->getCoalescedSet();
            if(coalescedSet != NULL) {
                assert(coalescedSet->isComplete()); // All previously coalesced sets should be complete
                for(MVMOperation* m : *coalescedSet) {
                    // If MVM is coalesced, include predecessors of all in the coalesced set
                    assert(m != NULL);
                    ProducerOperation* predecessor = m->getOperand(0);
                    findMVMPredecessors(predecessor, mvmPredecessors);
                    mvmPredecessors[op].insert(mvmPredecessors[predecessor].begin(), mvmPredecessors[predecessor].end());
                }
            } else {
                ProducerOperation* predecessor = mvm->getOperand(0);
                findMVMPredecessors(predecessor, mvmPredecessors);
                mvmPredecessors[op].insert(mvmPredecessors[predecessor].begin(), mvmPredecessors[predecessor].end());
            }
        } else if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(op)) {
            for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
                ProducerOperation* predecessor = consumer->getOperand(o);
                findMVMPredecessors(predecessor, mvmPredecessors);
                mvmPredecessors[op].insert(mvmPredecessors[predecessor].begin(), mvmPredecessors[predecessor].end());
                if(MVMOperation* mvmPred = dynamic_cast<MVMOperation*>(predecessor)) {
                    CoalescedMVMSet* coalescedSet = mvmPred->getCoalescedSet();
                    if(coalescedSet ==  NULL) {
                        // Only uncoalesced MVMs are interesting
                        mvmPredecessors[op].insert(mvmPred);
                    } else {
                        assert(coalescedSet->isComplete()); // All previously coalesced sets should be complete
                    }
                }
            }
        }
        if(TileMemoryReadOperation* read = dynamic_cast<TileMemoryReadOperation*>(op)) {
            for(unsigned int i = 0; i < read->numSrcs(); ++i) {
                TileMemoryWriteOperation* predecessor = read->getSrc(i);
                findMVMPredecessors(predecessor, mvmPredecessors);
                mvmPredecessors[op].insert(mvmPredecessors[predecessor].begin(), mvmPredecessors[predecessor].end());
            }
        }
        if(ReceiveOperation* recv = dynamic_cast<ReceiveOperation*>(op)) {
            SendOperation* predecessor = recv->getSrc();
            findMVMPredecessors(predecessor, mvmPredecessors);
            mvmPredecessors[op].insert(mvmPredecessors[predecessor].begin(), mvmPredecessors[predecessor].end());
        }
    }
}

void Coalescer::coalesceMVMPredecessors(Operation* op, std::set<Operation*>& isVisited, std::map<MVMOperation*, std::set<MVMOperation*>>& mvmPredecessorsOfMVMs, std::map<MVMOperation*, std::set<MVMOperation*>>& mvmSuccessorsOfMVMs) {
    if(!isVisited.count(op)) {
        // Visit nodes in reverse postorder (not necessary, but visiting in same order as linearization helps reduce register pressure)
        if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(op)) {
            for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
                ProducerOperation* predecessor = consumer->getOperand(o);
                coalesceMVMPredecessors(predecessor, isVisited, mvmPredecessorsOfMVMs, mvmSuccessorsOfMVMs);
            }
            if(MVMOperation* mvm = dynamic_cast<MVMOperation*>(consumer)) {
                if(mvm->getCoalescedSet() == NULL) {
                    // Find coalesced set to add to
                    std::vector<CoalescedMVMSet*> &coreCoalescedSets = coalescedMVMSets_[placer_->getPTile(mvm)*N_CORES_PER_TILE + placer_->getPCore(mvm)];
                    unsigned int pMVMU = placer_->getPMVMU(mvm);
                    CoalescedMVMSet* coalescedSet = NULL;
                    for(unsigned int coalescedSetIdx = 0; coalescedSetIdx < coreCoalescedSets.size(); ++coalescedSetIdx) {
                        coalescedSet = coreCoalescedSets[coalescedSetIdx];
                        if(!coalescedSet->usesPMVMU(pMVMU)) {
                            bool hasDataHazard = false;
                            for(MVMOperation* m : *coalescedSet) {
                                if(mvmPredecessorsOfMVMs[mvm].count(m) || mvmSuccessorsOfMVMs[mvm].count(m)) {
                                    hasDataHazard = true;
                                    break;
                                }
                            }
                            if(!hasDataHazard) {
                                break; // Candidate found
                            }
                        }
                        coalescedSet = NULL; // Candidate doesn't work
                    }
                    if(coalescedSet == NULL) {
                        // Create new coalesced set if none found
                        coalescedSet = new CoalescedMVMSet();
                        coreCoalescedSets.push_back(coalescedSet);
                    }
                    // Add to coalesced set and update dependence information
                    for(MVMOperation* m : *coalescedSet) {
                        if(m != NULL) {
                            // Make all predecessors of mvm predecessors of m and successors of m
                            for(MVMOperation* mvmPredecessor : mvmPredecessorsOfMVMs[mvm]) {
                                mvmPredecessorsOfMVMs[m].insert(mvmPredecessor);
                                mvmSuccessorsOfMVMs[mvmPredecessor].insert(m);
                                for(MVMOperation* mSuccessor : mvmSuccessorsOfMVMs[m]) {
                                    mvmPredecessorsOfMVMs[mSuccessor].insert(mvmPredecessor);
                                    mvmSuccessorsOfMVMs[mvmPredecessor].insert(mSuccessor);
                                }
                            }
                            // Make all predecessors of m predecessors of mvm and successors of mvm
                            for(MVMOperation* mPredecessor : mvmPredecessorsOfMVMs[m]) {
                                mvmPredecessorsOfMVMs[mvm].insert(mPredecessor);
                                mvmSuccessorsOfMVMs[mPredecessor].insert(mvm);
                                for(MVMOperation* mvmSuccessor : mvmSuccessorsOfMVMs[mvm]) {
                                    mvmPredecessorsOfMVMs[mvmSuccessor].insert(mPredecessor);
                                    mvmSuccessorsOfMVMs[mPredecessor].insert(mvmSuccessor);
                                }
                            }
                        }
                    }
                    coalescedSet->add(mvm, pMVMU);
                }
            }
        }
        if(TileMemoryReadOperation* read = dynamic_cast<TileMemoryReadOperation*>(op)) {
            for(unsigned int i = 0; i < read->numSrcs(); ++i) {
                TileMemoryWriteOperation* predecessor = read->getSrc(i);
                coalesceMVMPredecessors(predecessor, isVisited, mvmPredecessorsOfMVMs, mvmSuccessorsOfMVMs);
            }
        }
        if(ReceiveOperation* recv = dynamic_cast<ReceiveOperation*>(op)) {
            SendOperation* predecessor = recv->getSrc();
            coalesceMVMPredecessors(predecessor, isVisited, mvmPredecessorsOfMVMs, mvmSuccessorsOfMVMs);
        }
        isVisited.insert(op);
    }
}

void Coalescer::coalesceTrainingOperations() {

    coalescedTrainingOperationSets_.resize(placer_->getNPCores());

    // Find immediate training operation predecessors of each training operation
    std::map<TrainingMatrixOperation*, std::set<TrainingMatrixOperation*>> immediateTrainingOperationPredecessors;
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        if(TrainingMatrixOperation* trainOp = dynamic_cast<TrainingMatrixOperation*>(*it)) {
            findImmediateTrainingOperationPredecessors(trainOp, immediateTrainingOperationPredecessors[trainOp]);
        }
    }

    // Derive all training operation predecessors of each training operation
    std::map<TrainingMatrixOperation*, std::set<TrainingMatrixOperation*>> trainingOperationPredecessors;
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        if(TrainingMatrixOperation* trainOp = dynamic_cast<TrainingMatrixOperation*>(*it)) {
            findAllTrainingOperationPredecessors(trainOp, trainingOperationPredecessors[trainOp], immediateTrainingOperationPredecessors);
        }
    }

    // Derive all training operation successors for each training operation
    std::map<TrainingMatrixOperation*, std::set<TrainingMatrixOperation*>> trainingOperationSuccessors;
    for(auto it : trainingOperationPredecessors) {
        TrainingMatrixOperation* trainOp = it.first;
        for(TrainingMatrixOperation* predecessor : it.second) {
            trainingOperationSuccessors[predecessor].insert(trainOp);
        }
    }

    // Coalesce training operations (in linearization order)
    std::set<Operation*> isVisited;
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if(TrainingMatrixOperation* trainOp = dynamic_cast<TrainingMatrixOperation*>(op)) {
            if(trainOp->getOpType() == TrainingMatrixOperation::OUTER_PRODUCT) {
                coalesceTrainingOperationPredecessors(op, isVisited, trainingOperationPredecessors, trainingOperationSuccessors);
            }
        } else if(dynamic_cast<ReadOutputOperation*>(op)) {
            coalesceTrainingOperationPredecessors(op, isVisited, trainingOperationPredecessors, trainingOperationSuccessors);
        }
    }


}

void Coalescer::findImmediateTrainingOperationPredecessors(Operation* op, std::set<TrainingMatrixOperation*>& foundSet) {
    if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(op)) {
        for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
            ProducerOperation* predecessor = consumer->getOperand(o);
            if(TrainingMatrixOperation* trainOp = dynamic_cast<TrainingMatrixOperation*>(predecessor)) {
                foundSet.insert(trainOp);
            } else {
                findImmediateTrainingOperationPredecessors(predecessor, foundSet);
            }
        }
    }
    if(TileMemoryReadOperation* read = dynamic_cast<TileMemoryReadOperation*>(op)) {
        for(unsigned int i = 0; i < read->numSrcs(); ++i) {
            TileMemoryWriteOperation* predecessor = read->getSrc(i);
            findImmediateTrainingOperationPredecessors(predecessor, foundSet);
        }
    }
    if(ReceiveOperation* recv = dynamic_cast<ReceiveOperation*>(op)) {
        SendOperation* predecessor = recv->getSrc();
        findImmediateTrainingOperationPredecessors(predecessor, foundSet);
    }
}

void Coalescer::findAllTrainingOperationPredecessors(TrainingMatrixOperation* trainOp, std::set<TrainingMatrixOperation*>& foundSet, std::map<TrainingMatrixOperation*, std::set<TrainingMatrixOperation*>>& immediateTrainingOperationPredecessors) {
    for(TrainingMatrixOperation* predecessor : immediateTrainingOperationPredecessors[trainOp]) {
        foundSet.insert(predecessor);
        findAllTrainingOperationPredecessors(predecessor, foundSet, immediateTrainingOperationPredecessors);
    }
}

void Coalescer::coalesceTrainingOperationPredecessors(Operation* op, std::set<Operation*>& isVisited, std::map<TrainingMatrixOperation*, std::set<TrainingMatrixOperation*>>& trainingOperationPredecessors, std::map<TrainingMatrixOperation*, std::set<TrainingMatrixOperation*>>& trainingOperationSuccessors) {
    if(!isVisited.count(op)) {
        // Visit nodes in reverse postorder (not necessary, but visiting in same order as linearization helps reduce register pressure)
        if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(op)) {
            for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
                ProducerOperation* predecessor = consumer->getOperand(o);
                coalesceTrainingOperationPredecessors(predecessor, isVisited, trainingOperationPredecessors, trainingOperationSuccessors);
            }
            if(TrainingMatrixOperation* trainOp = dynamic_cast<TrainingMatrixOperation*>(consumer)) {
                if(trainOp->getCoalescedSet() == NULL) {
                    // Find coalesced set to add to
                    std::vector<CoalescedTrainingOperationSet*> &coreCoalescedSets = coalescedTrainingOperationSets_[placer_->getPTile(trainOp)*N_CORES_PER_TILE + placer_->getPCore(trainOp)];
                    unsigned int pMVMU = placer_->getPMVMU(trainOp);
                    TrainingMatrixOperation::OpType opType = trainOp->getOpType();
                    CoalescedTrainingOperationSet* coalescedSet = NULL;
                    for(unsigned int coalescedSetIdx = 0; coalescedSetIdx < coreCoalescedSets.size(); ++coalescedSetIdx) {
                        coalescedSet = coreCoalescedSets[coalescedSetIdx];
                        if(!coalescedSet->usesPMVMUForOp(pMVMU, opType)) {
                            bool hasDataHazard = false;
                            for(TrainingMatrixOperation* t : *coalescedSet) {
                                if(trainingOperationPredecessors[trainOp].count(t) || trainingOperationSuccessors[trainOp].count(t)) {
                                    hasDataHazard = true;
                                    break;
                                }
                            }
                            if(!hasDataHazard) {
                                break; // Candidate found
                            }
                        }
                        coalescedSet = NULL; // Candidate doesn't work
                    }
                    if(coalescedSet == NULL) {
                        // Create new coalesced set if none found
                        coalescedSet = new CoalescedTrainingOperationSet();
                        coreCoalescedSets.push_back(coalescedSet);
                    }
                    // Add to coalesced set and update dependence information
                    for(TrainingMatrixOperation* t : *coalescedSet) {
                        if(t != NULL) {
                            // Make all predecessors of trainOp predecessors of t and successors of t
                            for(TrainingMatrixOperation* trainOpPredecessor : trainingOperationPredecessors[trainOp]) {
                                trainingOperationPredecessors[t].insert(trainOpPredecessor);
                                trainingOperationSuccessors[trainOpPredecessor].insert(t);
                                for(TrainingMatrixOperation* tSuccessor : trainingOperationSuccessors[t]) {
                                    trainingOperationPredecessors[tSuccessor].insert(trainOpPredecessor);
                                    trainingOperationSuccessors[trainOpPredecessor].insert(tSuccessor);
                                }
                            }
                            // Make all predecessors of t predecessors of trainOp and successors of trainOp
                            for(TrainingMatrixOperation* tPredecessor : trainingOperationPredecessors[t]) {
                                trainingOperationPredecessors[trainOp].insert(tPredecessor);
                                trainingOperationSuccessors[tPredecessor].insert(trainOp);
                                for(TrainingMatrixOperation* trainOpSuccessor : trainingOperationSuccessors[trainOp]) {
                                    trainingOperationPredecessors[trainOpSuccessor].insert(tPredecessor);
                                    trainingOperationSuccessors[tPredecessor].insert(trainOpSuccessor);
                                }
                            }
                        }
                    }
                    coalescedSet->add(trainOp, pMVMU);
                }
            }
        }
        if(TileMemoryReadOperation* read = dynamic_cast<TileMemoryReadOperation*>(op)) {
            for(unsigned int i = 0; i < read->numSrcs(); ++i) {
                TileMemoryWriteOperation* predecessor = read->getSrc(i);
                coalesceTrainingOperationPredecessors(predecessor, isVisited, trainingOperationPredecessors, trainingOperationSuccessors);
            }
        }
        if(ReceiveOperation* recv = dynamic_cast<ReceiveOperation*>(op)) {
            SendOperation* predecessor = recv->getSrc();
            coalesceTrainingOperationPredecessors(predecessor, isVisited, trainingOperationPredecessors, trainingOperationSuccessors);
        }
        isVisited.insert(op);
    }

}

