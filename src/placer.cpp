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

#include "model.h"
#include "operations.h"
#include "partitioner.h"
#include "placer.h"

Placer::Placer(ModelImpl* model,Partitioner* partitioner)
    : model_(model), partitioner_(partitioner)
{
    assignPTiles();
    assignPCores();
    assignPMVMUs();
}

void Placer::assignPTiles() {

    // Assign virtual tiles to physical tiles
    nPTiles_ = partitioner_->getNVTiles();
    vtile2ptile_.resize(partitioner_->getNVTiles());
    vtile2ptile_[0] = 0; // Reserve tile 0 for sending inputs
    vtile2ptile_[1] = 1; // Reserve tile 1 for receiving outputs
    for(unsigned int vTile = 2; vTile < partitioner_->getNVTiles(); ++vTile) {
        // TODO: implement a more intelligent virtual to physical tile assignment
        unsigned int pTile = vTile;
        vtile2ptile_[vTile] = pTile;
    }

}

void Placer::assignPCores() {

    // Assign virtual cores to physical cores
    nPCores_ = nPTiles_*N_CORES_PER_TILE;
    vcore2pcore_.resize(partitioner_->getNVCores());
    std::vector<unsigned int> nPCoresPerPTile(nPTiles_);
    for(unsigned int vCore = 0; vCore < partitioner_->getNVCores(); ++vCore) {
        unsigned int vTile = partitioner_->getVTile(vCore);
        unsigned int pTile = vtile2ptile_[vTile];
        unsigned int pCore = nPCoresPerPTile[pTile]++;
        assert(pCore < N_CORES_PER_TILE);
        vcore2pcore_[vCore] = pCore;
    }

}

void Placer::assignPMVMUs() {

    // Assign virtual MVMUs to physical MVMUs
    unsigned int nMVMUSPerCore = (model_->getModelType() == ModelImpl::INFERENCE)?(N_CONSTANT_MVMUS_PER_CORE):(N_TRAINING_MVMUS_PER_CORE);
    nPMVMUs_ = nPCores_*nMVMUSPerCore;
    vmvmu2pmvmu_.resize(partitioner_->getNVMVMUs());
    std::vector<unsigned int> nPMVMUsPerPCore(nPCores_);
    for(unsigned int vMVMU = 0; vMVMU < partitioner_->getNVMVMUs(); ++vMVMU) {
        unsigned int vCore = partitioner_->getVCore(vMVMU);
        unsigned int pCore = vcore2pcore_[vCore];
        unsigned int vTile = partitioner_->getVTile(vCore);
        unsigned int pTile = vtile2ptile_[vTile];
        unsigned int pMVMU = nPMVMUsPerPCore[pTile*N_CORES_PER_TILE + pCore];
        nPMVMUsPerPCore[pTile*N_CORES_PER_TILE + pCore] += 1;
        assert(pMVMU < nMVMUSPerCore);
        vmvmu2pmvmu_[vMVMU] = pMVMU;
    }

}

unsigned int Placer::getPTile(ConstantMatrixTile* tile) {
    return vtile2ptile_[partitioner_->getVTile(tile)];
}

unsigned int Placer::getPCore(ConstantMatrixTile* tile) {
    return vcore2pcore_[partitioner_->getVCore(tile)];
}

unsigned int Placer::getPMVMU(ConstantMatrixTile* tile) {
    return vmvmu2pmvmu_[partitioner_->getVMVMU(tile)];
}

unsigned int Placer::getPTile(TrainingMatrixTile* tile) {
    return vtile2ptile_[partitioner_->getVTile(tile)];
}

unsigned int Placer::getPCore(TrainingMatrixTile* tile) {
    return vcore2pcore_[partitioner_->getVCore(tile)];
}

unsigned int Placer::getPMVMU(TrainingMatrixTile* tile) {
    return vmvmu2pmvmu_[partitioner_->getVMVMU(tile)];
}

unsigned int Placer::getPTile(Operation* op) {
    return vtile2ptile_[partitioner_->getVTile(op)];
}

unsigned int Placer::getPCore(Operation* op) {
    return vcore2pcore_[partitioner_->getVCore(op)];
}

unsigned int Placer::getPMVMU(Operation* op) {
    return vmvmu2pmvmu_[partitioner_->getVMVMU(op)];
}

std::string Placer::printAssignment(Operation* op) {
    std::stringstream ss;
    if(vmvmu2pmvmu_.size() > 0) {
        ss << "\npMVMU = " << getPMVMU(op);
    }
    if(vcore2pcore_.size() > 0) {
        ss << ", pCore = " << getPCore(op);
    }
    if(vtile2ptile_.size() > 0) {
        ss << ", pTile = " << getPTile(op);
    }
    return ss.str();
}

