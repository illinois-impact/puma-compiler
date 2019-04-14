/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>
#include <bitset>
#include <sstream>

#include "puma.h"

#include "linearizer.h"
#include "memalloc.h"
#include "model.h"
#include "operations.h"
#include "partitioner.h"
#include "placer.h"
#include "regalloc.h"

class CoreAllocator {

    private:

        std::bitset<REGISTERS_PER_CORE> memPool_;

    public:

        static const unsigned int OUT_OF_REGISTERS = REGISTERS_PER_CORE;

        unsigned int allocate(unsigned int size);
        void free(unsigned int pos, unsigned int size);

};

class SpillTracker {

    private:

        std::map<ProducerOperation*, StoreOperation*> producer2spill;
        std::map<ProducerOperation*, LoadOperation*> producer2reload;
        std::map<LoadOperation*, ProducerOperation*> reload2producer;

    public:

        bool isSpilled(ProducerOperation* producer) { return producer2spill.count(producer); }
        bool hasLiveNowReload(ProducerOperation* producer) { return producer2reload.count(producer); }
        bool isLiveNowReload(LoadOperation* load) { return reload2producer.count(load); }

        StoreOperation* getSpillOperation(ProducerOperation* producer);
        LoadOperation* getLiveNowReload(ProducerOperation* producer);
        ProducerOperation* getOriginalProducer(LoadOperation* load);

        void setSpillOperation(ProducerOperation* producer, StoreOperation* store);
        void setLiveNowReload(ProducerOperation* producer, LoadOperation* load);
        void killLiveNowReload(LoadOperation* load);

        std::map<ProducerOperation*, LoadOperation*>::iterator reloads_begin() { return producer2reload.begin(); }
        std::map<ProducerOperation*, LoadOperation*>::iterator reloads_end() { return producer2reload.end(); }

};

unsigned int CoreAllocator::allocate(unsigned int size) {
    for(unsigned int i = 0; i <= REGISTER_FILE_SIZE - size; ++i) {
        unsigned int j;
        for(j = i; j < i + size; ++j) {
            if(memPool_[j]) {
                break;
            }
        }
        if(j == i + size) {
            for(unsigned int k = i; k < j; ++k) {
                memPool_.set(k);
            }
            return REGISTER_FILE_START_ADDRESS + i;
        } else {
            i = j;
        }
    }
    return OUT_OF_REGISTERS;
}

void CoreAllocator::free(unsigned int reg, unsigned int size) {
    unsigned int pos = reg - REGISTER_FILE_START_ADDRESS;
    for(unsigned int i = pos; i < pos + size; ++i) {
        assert(memPool_[i] && "Attempt to free unallocated registers!");
        memPool_.reset(i);
    }
}

StoreOperation* SpillTracker::getSpillOperation(ProducerOperation* producer) {
    assert(isSpilled(producer));
    return producer2spill[producer];
}

LoadOperation* SpillTracker::getLiveNowReload(ProducerOperation* producer) {
    assert(hasLiveNowReload(producer));
    return producer2reload[producer];
}

ProducerOperation* SpillTracker::getOriginalProducer(LoadOperation* load) {
    assert(isLiveNowReload(load));
    return reload2producer[load];
}

void SpillTracker::setSpillOperation(ProducerOperation* producer, StoreOperation* store) {
    assert(!producer2spill.count(producer) && "Register allocation error: spilling a register that has already been spilled!");
    producer2spill[producer] = store;
}

void SpillTracker::setLiveNowReload(ProducerOperation* producer, LoadOperation* load) {
    assert(!hasLiveNowReload(producer) && "Register allocation error: reloading a spilled register that has already been reloaded!");
    producer2reload[producer] = load;
    reload2producer[load] = producer;
}

void SpillTracker::killLiveNowReload(LoadOperation* load) {
    assert(isLiveNowReload(load));
    ProducerOperation* producer = reload2producer[load];
    producer2reload.erase(producer);
    reload2producer.erase(load);
}

RegisterAllocator::RegisterAllocator(ModelImpl* model, Partitioner* partitioner, Placer* placer, MemoryAllocator* memoryAllocator, Linearizer* linearizer)
 : model_(model), partitioner_(partitioner), placer_(placer), memoryAllocator_(memoryAllocator), linearizer_(linearizer)
{
    registerAllocation();
}

bool RegisterAllocator::isRegisterAssigned(ProducerOperation* producer) {
    return op2reg_.count(producer);
}

void RegisterAllocator::assignRegister(ProducerOperation* producer, unsigned int reg) {
    assert(!isRegisterAssigned(producer) && "Cannot reassign register");
    op2reg_[producer] = reg;
}

void RegisterAllocator::assignReservedInputRegister(ProducerOperation* producer) {
    assert(!writesToReservedOutputRegister(producer) && "Cannot assign reserved input registers to matrix operations that write to reserved output registers!");
    assert(producer->numUsers() == 1 && "Producer serving a matrix operation can only have one user");
    ConsumerOperation* consumer = *(producer->user_begin());
    unsigned int reg;
    if(MVMOperation* mvm = dynamic_cast<MVMOperation*>(consumer)) {
        reg = INPUT_REGISTERS_START_ADDRESS + placer_->getPMVMU(mvm)*MVMU_DIM;
    } else if(TrainingMatrixOperation* trainOp = dynamic_cast<TrainingMatrixOperation*>(consumer)) {
        switch(trainOp->getOpType()) {
            case TrainingMatrixOperation::MVM:
                reg = INPUT_REGISTERS_START_ADDRESS + placer_->getPMVMU(trainOp)*N_TRAINING_OPERATIONS*MVMU_DIM;
                break;
            case TrainingMatrixOperation::MVM_TRANSPOSE:
                reg = INPUT_REGISTERS_START_ADDRESS + (placer_->getPMVMU(trainOp)*N_TRAINING_OPERATIONS + 1)*MVMU_DIM;
                break;
            case TrainingMatrixOperation::OUTER_PRODUCT:
            {
                if(producer == consumer->getOperand(0)) {
                    reg = INPUT_REGISTERS_START_ADDRESS + (placer_->getPMVMU(trainOp)*N_TRAINING_OPERATIONS + 2)*MVMU_DIM;
                } else if(producer == consumer->getOperand(1)) {
                    // NOTE: In training mode, some output registers are used as the second input register to the outer product operation
                    reg = OUTPUT_REGISTERS_START_ADDRESS + (placer_->getPMVMU(trainOp)*N_TRAINING_OPERATIONS + 2)*MVMU_DIM;
                } else {
                    assert(0 && "Impossible case!");
                }
                break;
            }
            default: assert(0 && "Impossible case!");
        }
    } else {
        assert(0 && "Cannot assign reserved input register to producer that doesn't feed a matrix operation");
    }
    assignRegister(producer, reg);
}

void RegisterAllocator::assignReservedOutputRegister(ProducerOperation* producer) {
    assert(writesToReservedOutputRegister(producer) && "Cannot assign reserved output registers to non-matrix operations");
    unsigned int reg;
    if(MVMOperation* mvm = dynamic_cast<MVMOperation*>(producer)) {
        reg = OUTPUT_REGISTERS_START_ADDRESS + placer_->getPMVMU(mvm)*MVMU_DIM;
    } else if(TrainingMatrixOperation* trainOp = dynamic_cast<TrainingMatrixOperation*>(producer)) {
        switch(trainOp->getOpType()) {
            case TrainingMatrixOperation::MVM:
                reg = OUTPUT_REGISTERS_START_ADDRESS + placer_->getPMVMU(trainOp)*N_TRAINING_OPERATIONS*MVMU_DIM;
                break;
            case TrainingMatrixOperation::MVM_TRANSPOSE:
                reg = OUTPUT_REGISTERS_START_ADDRESS + (placer_->getPMVMU(trainOp)*N_TRAINING_OPERATIONS + 1)*MVMU_DIM;
                break;
            // NOTE: Outer product operations do not write to reserved output registers, they read from them
            default: assert(0 && "Impossible case!");
        }
    } else {
        assert(0 && "Cannot assign reserved output register to producer that is not a matrix operation");
    }
    assignRegister(producer, reg);
}

bool RegisterAllocator::readsFromReservedInputRegister(ConsumerOperation* consumer) {
    return (dynamic_cast<MVMOperation*>(consumer) != NULL)
            || (dynamic_cast<TrainingMatrixOperation*>(consumer) != NULL);
}

bool RegisterAllocator::writesToReservedOutputRegister(ProducerOperation* producer) {
    return (dynamic_cast<MVMOperation*>(producer) != NULL)
            || ((dynamic_cast<TrainingMatrixOperation*>(producer) != NULL)
                && !producerDoesNotWriteToRegister(producer));
}

bool RegisterAllocator::producerDoesNotWriteToRegister(ProducerOperation* producer) {
    if(TrainingMatrixOperation* trainOp = dynamic_cast<TrainingMatrixOperation*>(producer)) {
        if(trainOp->getOpType() == TrainingMatrixOperation::OUTER_PRODUCT) {
            // NOTE: Outer products are declared as producer operations so they can be coalesced with other training operations, but they do not write to any registers
            return true;
        }
    }
    return false;
}

unsigned int RegisterAllocator::getRegister(ProducerOperation* producer) {
    assert(isRegisterAssigned(producer) && "Register has not been assigned!");
    return op2reg_[producer];
}

void RegisterAllocator::registerAllocation() {

    // Allocate registers
    for(unsigned int pTile = 0; pTile < placer_->getNPTiles(); ++pTile) {
        for(unsigned int pCore = 0; pCore < N_CORES_PER_TILE; ++pCore) {
            allocateReservedInputRegisters(pTile, pCore);
            allocateReservedOutputRegisters(pTile, pCore);
            allocateDataRegisters(pTile, pCore);
        }
    }

}

void RegisterAllocator::allocateReservedInputRegisters(unsigned int pTile, unsigned int pCore) {
    // Assign reserved input registers and ensure no overlap in live ranges
    std::set<ProducerOperation*> liveNow;
    std::list<CoreOperation*>& coreOperationList = linearizer_->getCoreOperationList(pTile, pCore);
    for(auto op = coreOperationList.rbegin(); op != coreOperationList.rend(); ++op) {
        if(ProducerOperation* producer = dynamic_cast<ProducerOperation*>(*op)) {
            liveNow.erase(producer);
        }
        if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(*op)) {
            if(readsFromReservedInputRegister(consumer)) {
                for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
                    ProducerOperation* producer = consumer->getOperand(o);
                    if(!liveNow.count(producer)) {
                        liveNow.insert(producer);
                        assignReservedInputRegister(producer);
                        for(ProducerOperation* p : liveNow) {
                            if(p != producer && getRegister(p) == getRegister(producer)) {
                                // NOTE: The linearizer ensures that there are no live range conflicts by placing matrix operation operands immediately before they are consumed
                                assert(0 && "Register allocation error: conflict detected in live ranges of operations using the same reserved input registers!");
                            }
                        }
                    }
                }
            }
        }
    }
}

void RegisterAllocator::allocateReservedOutputRegisters(unsigned int pTile, unsigned int pCore) {
    // Assign reserved output registers and ensure no overlap in live ranges
    std::set<ProducerOperation*> liveNow;
    std::list<CoreOperation*>& coreOperationList = linearizer_->getCoreOperationList(pTile, pCore);
    for(auto op = coreOperationList.rbegin(); op != coreOperationList.rend(); ++op) {
        if(ProducerOperation* producer = dynamic_cast<ProducerOperation*>(*op)) {
            liveNow.erase(producer);
        }
        if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(*op)) {
            for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
                ProducerOperation* producer = consumer->getOperand(o);
                if(writesToReservedOutputRegister(producer)) {
                    if(!liveNow.count(producer)) {
                        liveNow.insert(producer);
                        assignReservedOutputRegister(producer);
                        for(ProducerOperation* p : liveNow) {
                            if(p != producer && getRegister(p) == getRegister(producer)) {
                                // NOTE: The linearizer ensures that there are no live range conflicts by placing matrix operation consumers immediately after they are produced
                                assert(0 && "Register allocation error: conflict detected in live ranges of operations using the same reserved output registers!");
                            }
                        }
                    }
                }
            }
        }
    }
}

void RegisterAllocator::allocateDataRegisters(unsigned int pTile, unsigned int pCore) {

    // Live range analysis
    std::list<CoreOperation*>& coreOperationList = linearizer_->getCoreOperationList(pTile, pCore);
    std::map<Operation*, std::set<ProducerOperation*>> liveIn;
    Operation* nextOp = NULL;
    for(auto op = coreOperationList.rbegin(); op != coreOperationList.rend(); ++op) {

        // Clone the live in set of the next operation
        liveIn[*op].insert(liveIn[nextOp].begin(), liveIn[nextOp].end());

        // Remove operations produced by this operation
        if(ProducerOperation* producer = dynamic_cast<ProducerOperation*>(*op)) {
            liveIn[*op].erase(producer);
        }

        // Add operations consumed by the operation
        if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(*op)) {
            if(!readsFromReservedInputRegister(consumer)) {
                for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
                    ProducerOperation* producer = consumer->getOperand(o);
                    if(!writesToReservedOutputRegister(producer)) {
                        liveIn[*op].insert(producer);
                    }
                }
            }
        }

        nextOp = *op;

    }

    // Allocate data registers
    CoreAllocator allocator;
    SpillTracker spillTracker;
    std::set<ProducerOperation*> liveNow;
    unsigned int spillAddressReg = allocator.allocate(1);
    for(auto op = coreOperationList.begin(); op != coreOperationList.end(); ++op) {

        auto next = op; ++next;
        Operation* nextOp = (next != coreOperationList.end())?(*next):(NULL);
        std::set<ProducerOperation*>& liveOut = liveIn[nextOp];

        // Process operands
        if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(*op)) {
            if(!readsFromReservedInputRegister(consumer)) {

                // Make sure all operands are available
                for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
                    ProducerOperation* producer = consumer->getOperand(o);
                    if(!writesToReservedOutputRegister(producer)) {
                        if(liveNow.count(producer) || spillTracker.isLiveNowReload(dynamic_cast<LoadOperation*>(producer))) {
                            numUnspilledRegAccesses_ += producer->length();
                        } else {
                            // Reload operands that have been spilled
                            assert(spillTracker.isSpilled(producer));
                            if(spillTracker.hasLiveNowReload(producer)) {
                                // If already reloaded, reuse reload
                                numUnspilledRegAccesses_ += producer->length();
                                LoadOperation* load = spillTracker.getLiveNowReload(producer);
                                consumer->replaceOperand(producer, load);
                            } else {
                                // Reload from spilled register
                                numSpilledRegAccesses_ += producer->length();
                                StoreOperation* spillOp = spillTracker.getSpillOperation(producer);
                                SetImmediateOperation* seti = new SetImmediateOperation(model_, memoryAllocator_->getTileMemoryAddress(spillOp));
                                partitioner_->cloneAssignment(producer, seti);
                                assignRegister(seti, spillAddressReg);
                                LoadOperation* load = new LoadOperation(model_, spillOp);
                                numLoadsFromSpilling_ += load->length();
                                load->addTileMemoryAddressOperand(seti);
                                partitioner_->cloneAssignment(producer, load);
                                unsigned int reg = allocateRegistersWithSpilling(load->length(), allocator, liveNow, spillTracker, spillAddressReg, coreOperationList, op);
                                assignRegister(load, reg);
                                consumer->replaceOperand(producer, load);
                                coreOperationList.insert(op, seti);
                                coreOperationList.insert(op, load);
                                spillTracker.setLiveNowReload(producer, load);
                            }
                        }
                    }
                }

                // Free registers for operands that are no longer live
                for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
                    ProducerOperation* producer = consumer->getOperand(o);
                    if(!writesToReservedOutputRegister(producer)) {
                        if(liveNow.count(producer)) {
                            if(!liveOut.count(producer)) {
                                liveNow.erase(producer);
                                allocator.free(getRegister(producer), producer->length());
                            }
                        } else if(LoadOperation* load = dynamic_cast<LoadOperation*>(producer)) {
                            assert(spillTracker.isLiveNowReload(load));
                            ProducerOperation* originalProducer = spillTracker.getOriginalProducer(load);
                            if(!liveOut.count(originalProducer)) {
                                spillTracker.killLiveNowReload(load);
                                allocator.free(getRegister(load), load->length());
                            }
                        } else {
                            assert(0 && "Operand must either be a live operation or a spilled register load!");
                        }
                    }
                }

            }
        }

        // Allocate register for new operation
        if(ProducerOperation* producer = dynamic_cast<ProducerOperation*>(*op)) {
            assert(!liveIn[*op].count(producer));
            if(liveOut.count(producer)) {
                unsigned int reg = allocateRegistersWithSpilling(producer->length(), allocator, liveNow, spillTracker, spillAddressReg, coreOperationList, op);
                assignRegister(producer, reg);
                liveNow.insert(producer);
            } else {
                // Producer already assigned to a reserved input or output register
                assert(isRegisterAssigned(producer) || producerDoesNotWriteToRegister(producer));
            }
        }

    }

}

unsigned int RegisterAllocator::allocateRegistersWithSpilling(unsigned int length, CoreAllocator& allocator, std::set<ProducerOperation*>& liveNow, SpillTracker& spillTracker, unsigned int spillAddressReg, std::list<CoreOperation*>& coreOperationList, std::list<CoreOperation*>::iterator& op) {

    // TODO: Better heuristic for which is the best register to free (e.g., the one which will be used the latest into the future)
    ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(*op);
    unsigned int reg = allocator.allocate(length);
    if(reg != CoreAllocator::OUT_OF_REGISTERS) {
        return reg;
    } else {
        // First try to free registers by killing live reloads that are not used by this operation
        for(auto killCandidate = spillTracker.reloads_begin(); killCandidate != spillTracker.reloads_end(); ++killCandidate) {
            ProducerOperation* producerToKill = killCandidate->first;
            LoadOperation* reloadToKill = killCandidate->second;
            if(consumer == NULL || !consumer->uses(producerToKill) && !consumer->uses(reloadToKill)) {
                spillTracker.killLiveNowReload(reloadToKill);
                allocator.free(getRegister(reloadToKill), reloadToKill->length());
                reg = allocator.allocate(length);
                if(reg != CoreAllocator::OUT_OF_REGISTERS) {
                    return reg;
                }
            }
        }
        // If unable to kill enough reloads, then spill live operations that are not used by this operation
        for(ProducerOperation* spillCandidate : liveNow) {
            if(consumer == NULL || !consumer->uses(spillCandidate)) {
                unsigned int address = memoryAllocator_->memalloc(partitioner_->getVTile(spillCandidate), spillCandidate->length());
                SetImmediateOperation* setiStore = new SetImmediateOperation(model_, address);
                partitioner_->cloneAssignment(spillCandidate, setiStore);
                assignRegister(setiStore, spillAddressReg);
                StoreOperation* store = new StoreOperation(model_, spillCandidate);
                numStoresFromSpilling_ += store->length();
                partitioner_->cloneAssignment(spillCandidate, store);
                memoryAllocator_->assignTileMemoryAddress(store, address);
                store->addTileMemoryAddressOperand(setiStore);
                coreOperationList.insert(op, setiStore);
                coreOperationList.insert(op, store);
                liveNow.erase(spillCandidate);
                spillTracker.setSpillOperation(spillCandidate, store);
                allocator.free(getRegister(spillCandidate), spillCandidate->length());
                reg = allocator.allocate(length);
                if(reg != CoreAllocator::OUT_OF_REGISTERS) {
                    return reg;
                }
            }
        }
        // If unable to spill enough live operations, then fail
        assert(0 && "Register allocation error: cannot find enough registers to spill!");
    }

}

void RegisterAllocator::printReport(std::ofstream& report) {
    report << "# load bytes from spilling = " << numLoadsFromSpilling_ << std::endl;
    report << "# store bytes from spilling = " << numStoresFromSpilling_ << std::endl;
    report << "# load + store bytes from spilling = " << numLoadsFromSpilling_ + numStoresFromSpilling_ << std::endl;
    report << "# unspilled register accesses = " << numUnspilledRegAccesses_ << std::endl;
    report << "# spilled register accesses = " << numSpilledRegAccesses_ << std::endl;
    report << "% spilled register accesses = " << 100.0*numSpilledRegAccesses_/(numSpilledRegAccesses_ + numUnspilledRegAccesses_) << "%" << std::endl;
}

std::string RegisterAllocator::printAssignment(Operation* op) {
    std::stringstream ss;
    if(ProducerOperation* producer = dynamic_cast<ProducerOperation*>(op)) {
        if(isRegisterAssigned(producer)) {
            ss << "\nregister = " << getRegister(producer);
        }
    }
    return ss.str();
}

