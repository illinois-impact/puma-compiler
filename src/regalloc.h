/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <fstream>
#include <list>
#include <map>
#include <string>

#include "common.h"

class RegisterAllocator {

    private:

        ModelImpl* model_;
        Partitioner* partitioner_;
        Placer* placer_;
        MemoryAllocator* memoryAllocator_;
        Linearizer* linearizer_;

        std::map<ProducerOperation*, unsigned int> op2reg_;

        unsigned int numLoadsFromSpilling_ = 0;
        unsigned int numStoresFromSpilling_ = 0;
        unsigned int numUnspilledRegAccesses_ = 0;
        unsigned int numSpilledRegAccesses_ = 0;

        void assignRegister(ProducerOperation* producer, unsigned int reg);
        void assignReservedInputRegister(ProducerOperation* producer);
        void assignReservedOutputRegister(ProducerOperation* producer);
        bool readsFromReservedInputRegister(ConsumerOperation* consumer);
        bool writesToReservedOutputRegister(ProducerOperation* producer);
        bool producerDoesNotWriteToRegister(ProducerOperation* producer);
        bool isRegisterAssigned(ProducerOperation* producer);
        void registerAllocation();
        void allocateReservedInputRegisters(unsigned int pTile, unsigned int pCore);
        void allocateReservedOutputRegisters(unsigned int pTile, unsigned int pCore);
        void allocateDataRegisters(unsigned int pTile, unsigned int pCore);
        unsigned int allocateRegistersWithSpilling(unsigned int length, CoreAllocator& allocator, std::set<ProducerOperation*>& liveNow, SpillTracker& spillTracker, unsigned int spillAddressReg, std::list<CoreOperation*>& coreOperationList, std::list<CoreOperation*>::iterator& op);

    public:

        RegisterAllocator(ModelImpl* model, Partitioner* partitioner, Placer* placer, MemoryAllocator* memoryAllocator, Linearizer* linearizer);

        unsigned int getRegister(ProducerOperation* producer);

        void printReport(std::ofstream& report);
        std::string printAssignment(Operation* op);

};

