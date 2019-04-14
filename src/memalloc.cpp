/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>
#include <sstream>

#include "puma.h"

#include "memalloc.h"
#include "model.h"
#include "operations.h"
#include "partitioner.h"

MemoryAllocator::MemoryAllocator(ModelImpl* model, Partitioner* partitioner)
    : model_(model), partitioner_(partitioner)
{
    memoryAllocation();
}

void MemoryAllocator::memoryAllocation() {

    // Tile memory allocation
    vTileAvailableMemory_.resize(partitioner_->getNVTiles());
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if(TileMemoryWriteOperation* write = dynamic_cast<TileMemoryWriteOperation*>(op)) {
            // FIXME: Receives used by the same read output operation on tile 1 should be assigned the same memory location
            unsigned int address = memalloc(partitioner_->getVTile(write), write->length());
            assignTileMemoryAddress(write, address);
            if(StoreOperation* store = dynamic_cast<StoreOperation*>(write)) {
                SetImmediateOperation* seti = new SetImmediateOperation(model_, address);
                partitioner_->cloneAssignment(store, seti);
                store->addTileMemoryAddressOperand(seti);
            }
            for(auto u = write->user_begin(); u != write->user_end(); ++u) {
                TileMemoryReadOperation* read = *u;
                if(LoadOperation* load = dynamic_cast<LoadOperation*>(read)) {
                    SetImmediateOperation* seti = new SetImmediateOperation(model_, address);
                    partitioner_->cloneAssignment(load, seti);
                    load->addTileMemoryAddressOperand(seti);
                }
            }
        }
    }

}

bool MemoryAllocator::isTileMemoryAddressAssigned(TileMemoryWriteOperation* op) {
    return op2mem_.count(op);
}

void MemoryAllocator::assignTileMemoryAddress(TileMemoryWriteOperation* op, unsigned int address) {
    assert(!isTileMemoryAddressAssigned(op) && "Cannot reassign tile memory address");
    op2mem_[op] = address;
}

unsigned int MemoryAllocator::getTileMemoryAddress(TileMemoryWriteOperation* op) {
    assert(isTileMemoryAddressAssigned(op) && "Tile memory address has not been assigned");
    return op2mem_[op];
}

unsigned int MemoryAllocator::memalloc(unsigned int vTile, unsigned int size) {
    unsigned int address = vTileAvailableMemory_[vTile];
    vTileAvailableMemory_[vTile] += size;
    return address;
}

std::string MemoryAllocator::printAssignment(Operation* op) {
    std::stringstream ss;
    if(TileMemoryWriteOperation* write = dynamic_cast<TileMemoryWriteOperation*>(op)) {
        if(isTileMemoryAddressAssigned(write)) {
            ss << "\ntileMemoryAddress = " << getTileMemoryAddress(write);
        }
    }
    return ss.str();
}

