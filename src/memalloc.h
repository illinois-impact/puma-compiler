/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <map>

#include "common.h"

class MemoryAllocator {

    private:

        ModelImpl* model_;
        Partitioner* partitioner_;

        std::map<TileMemoryWriteOperation*, unsigned int> op2mem_;
        std::vector<unsigned int> vTileAvailableMemory_;

        bool isTileMemoryAddressAssigned(TileMemoryWriteOperation* op);
        void memoryAllocation();

    public:

        MemoryAllocator(ModelImpl* model, Partitioner* partitioner);

        void assignTileMemoryAddress(TileMemoryWriteOperation* op, unsigned int address);
        unsigned int getTileMemoryAddress(TileMemoryWriteOperation* op);
        unsigned int memalloc(unsigned int vTile, unsigned int size);

        std::string printAssignment(Operation* op);

};

