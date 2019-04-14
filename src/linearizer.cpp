/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>

#include "puma.h"

#include "linearizer.h"
#include "model.h"
#include "operations.h"
#include "partitioner.h"
#include "placer.h"

Linearizer::Linearizer(ModelImpl* model, Partitioner* partitioner, Placer* placer)
    : model_(model), partitioner_(partitioner), placer_(placer), coreOperationLists_(placer_->getNPCores()), tileOperationLists_(placer_->getNPTiles())
{
    linearize();
}

void Linearizer::linearize() {

    // Begin traversal from operations that output final results, namely matrix update operations and output operations
    std::set<Operation*> isVisited;
    std::set<Operation*> wasAddedEarly;
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if(TrainingMatrixOperation* trainOp = dynamic_cast<TrainingMatrixOperation*>(op)) {
            if(trainOp->getOpType() == TrainingMatrixOperation::OUTER_PRODUCT) {
                linearizeWithPredecessors(op, isVisited, wasAddedEarly);
            }
        } else if(dynamic_cast<ReadOutputOperation*>(op)) {
            linearizeWithPredecessors(op, isVisited, wasAddedEarly);
        }
    }

}

void Linearizer::linearizeWithPredecessors(Operation* op, std::set<Operation*>& isVisited, std::set<Operation*>& wasAddedEarly, bool addSelf) {
    /*
     * Linearization follows the following guidelines:
     *  (1) All predecessors of an operation are executed before the operation to ensure that data-dependeces are satisfied (reverse postorder achieves this)
     *  (2) Proritize depth over breadth to reduce the span of live ranges to minimize data register spilling (reverse postorder achieves this)
     *  (3) Consume matrix operation inputs immediately after they are produced to eliminate reserved input register live range conflicts
     *  (4) Consume matrix operation outputs immediately after they are produced to eliminate reserved output register live range conflicts
     */
    if(!isVisited.count(op)) {
        if(MVMOperation* mvm = dynamic_cast<MVMOperation*>(op)) {
            assert(addSelf); // addSelf is only false for operations that feed matrix operations, and matrix operations can't feed other matrix operations
            CoalescedMVMSet* coalescedSet = mvm->getCoalescedSet();
            if(coalescedSet != NULL) {
                // If MVM is coalesced, visit predecessors of all coalesced MVMs together
                for(MVMOperation* m : *coalescedSet) {
                    if(m != NULL) {
                        assert(m->numOperands() == 1);
                        linearizeWithPredecessors(m->getOperand(0), isVisited, wasAddedEarly, false); // Do not add inputs to instruction list yet
                    }
                }
                // Add inputs immediately before they are consumed
                for(MVMOperation* m : *coalescedSet) {
                    if(m != NULL) {
                        assert(m->numOperands() == 1);
                        ProducerOperation* operand = m->getOperand(0);
                        if(wasAddedEarly.count(operand)) {
                            // If an operand's predecessor is a matrix operation, it's predecessor will add it early. In this case, we add a copy operation.
                            CopyOperation* copy = new CopyOperation(model_, operand);
                            partitioner_->cloneAssignment(operand, copy);
                            m->replaceOperand(operand, copy);
                            operand = copy;
                        }
                        addToList(operand, isVisited);
                    }
                }
                // Add all MVMs in the coalesced set
                for(MVMOperation* m : *coalescedSet) {
                    if(m != NULL) {
                        addToList(m, isVisited);
                    }
                }
                // Consume outputs immediately after they are produced
                for(MVMOperation* m : *coalescedSet) {
                    if(m != NULL) {
                        addConsumersToList(m, isVisited, wasAddedEarly);
                    }
                }
            } else {
                assert(mvm->numOperands() == 1);
                linearizeWithPredecessors(mvm->getOperand(0), isVisited, wasAddedEarly);
                addToList(mvm, isVisited);
                // Consume outputs immediately after they are produced
                addConsumersToList(mvm, isVisited, wasAddedEarly);
            }
        } else if(TrainingMatrixOperation* trainOp = dynamic_cast<TrainingMatrixOperation*>(op)) {
            assert(addSelf); // addSelf is only false for operations that feed matrix operations, and matrix operations can't feed other matrix operations
            CoalescedTrainingOperationSet* coalescedSet = trainOp->getCoalescedSet();
            if(coalescedSet != NULL) {
                // If training matrix operation is coalesced, visit predecessors of all coalesced operations together
                for(TrainingMatrixOperation* t : *coalescedSet) {
                    if(t != NULL) {
                        for(unsigned int o = 0; o < t->numOperands(); ++o) {
                            linearizeWithPredecessors(t->getOperand(o), isVisited, wasAddedEarly, false); // Do not add inputs to instruction list yet
                        }
                    }
                }
                // Add inputs immediately before they are consumed
                for(TrainingMatrixOperation* t : *coalescedSet) {
                    if(t != NULL) {
                        for(unsigned int o = 0; o < t->numOperands(); ++o) {
                            ProducerOperation* operand = t->getOperand(o);
                            if(wasAddedEarly.count(operand)) {
                                // If an operand's predecessor is a matrix operation, it's predecessor will add it early. In this case, we add a copy operation.
                                CopyOperation* copy = new CopyOperation(model_, operand);
                                partitioner_->cloneAssignment(operand, copy);
                                t->replaceOperand(operand, copy);
                                operand = copy;
                            }
                            addToList(operand, isVisited);
                        }
                    }
                }
                // Add all matrix operations in the coalesced set
                for(TrainingMatrixOperation* t : *coalescedSet) {
                    if(t != NULL) {
                        addToList(t, isVisited);
                    }
                }
                // Consume outputs immediately after they are produced
                for(TrainingMatrixOperation* t : *coalescedSet) {
                    if(t != NULL) {
                        addConsumersToList(t, isVisited, wasAddedEarly);
                    }
                }
            } else {
                for(unsigned int o = 0; o < trainOp->numOperands(); ++o) {
                    linearizeWithPredecessors(trainOp->getOperand(o), isVisited, wasAddedEarly);
                }
                addToList(trainOp, isVisited);
                // Consume outputs immediately after they are produced
                addConsumersToList(trainOp, isVisited, wasAddedEarly);
            }
        } else {
            if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(op)) {
                for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
                    linearizeWithPredecessors(consumer->getOperand(o), isVisited, wasAddedEarly);
                }
            }
            if(TileMemoryReadOperation* read = dynamic_cast<TileMemoryReadOperation*>(op)) {
                for(unsigned int i = 0; i < read->numSrcs(); ++i) {
                    linearizeWithPredecessors(read->getSrc(i), isVisited, wasAddedEarly);
                }
                assert(!wasAddedEarly.count(read));
            }
            if(ReceiveOperation* recv = dynamic_cast<ReceiveOperation*>(op)) {
                linearizeWithPredecessors(recv->getSrc(), isVisited, wasAddedEarly);
                assert(!wasAddedEarly.count(recv));
            }
            if(addSelf) {
                if(!wasAddedEarly.count(op)) { // Do not add a consumer operation if it was added early by a predecesor matrix operation
                    addToList(op, isVisited);
                }
            }
        }
    }
}

void Linearizer::addToList(Operation* op, std::set<Operation*>& isVisited) {
    assert(!isVisited.count(op));
    if(CoreOperation* coreOp = dynamic_cast<CoreOperation*>(op)) {
        getCoreOperationList(placer_->getPTile(coreOp), placer_->getPCore(coreOp)).push_back(coreOp);
    }
    if(TileOperation* tileOp = dynamic_cast<TileOperation*>(op)) {
        getTileOperationList(placer_->getPTile(tileOp)).push_back(tileOp);
    }
    isVisited.insert(op);
}

void Linearizer::addConsumersToList(ProducerOperation* producer, std::set<Operation*>& isVisited, std::set<Operation*>& wasAddedEarly) {
    bool allConsumersCanBeAdded = true;
    for(auto u = producer->user_begin(); u != producer->user_end(); ++u) {
        ConsumerOperation* consumer = *u;
        bool consumerCanBeAdded = true;
        for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
            if(!isVisited.count(consumer->getOperand(o))) {
                consumerCanBeAdded = false;
                break;
            }
        }
        if(!consumerCanBeAdded) {
            allConsumersCanBeAdded = false;
            break;
        }
    }
    if(allConsumersCanBeAdded) {
        for(auto u = producer->user_begin(); u != producer->user_end(); ++u) {
            ConsumerOperation* consumer = *u;
            if(!wasAddedEarly.count(consumer)) {
                addToList(consumer, isVisited);
                wasAddedEarly.insert(consumer);
            }
        }
    } else {
        CopyOperation* copy = new CopyOperation(model_, producer);
        partitioner_->cloneAssignment(producer, copy);
        addToList(copy, isVisited);
        for(auto u = producer->user_begin(); u != producer->user_end(); ) {
            ConsumerOperation* consumer = *u;
            ++u; // replaceOperand may remove consumer from producer's users
            if(consumer != copy) {
                consumer->replaceOperand(producer, copy);
            }
        }
    }
}

std::list<CoreOperation*>& Linearizer::getCoreOperationList(unsigned int pTile, unsigned int pCore) {
    return coreOperationLists_[pTile*N_CORES_PER_TILE + pCore];
}

std::list<TileOperation*>& Linearizer::getTileOperationList(unsigned int pTile) {
    return tileOperationLists_[pTile];
}

