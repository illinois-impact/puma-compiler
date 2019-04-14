/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "common.h"

class Placer {

    private:

        ModelImpl* model_;
        Partitioner* partitioner_;

        unsigned int nPTiles_;
        unsigned int nPCores_;
        unsigned int nPMVMUs_;

        std::vector<unsigned int> vtile2ptile_;
        std::vector<unsigned int> vcore2pcore_;
        std::vector<unsigned int> vmvmu2pmvmu_;

        void assignPTiles();
        void assignPCores();
        void assignPMVMUs();

    public:

        Placer(ModelImpl* model, Partitioner* partitioner);

        unsigned int getNPMVMUs() { return nPMVMUs_; }
        unsigned int getNPCores() { return nPCores_; }
        unsigned int getNPTiles() { return nPTiles_; }
        unsigned int getPMVMU(ConstantMatrixTile* tile);
        unsigned int getPTile(ConstantMatrixTile* tile);
        unsigned int getPCore(ConstantMatrixTile* tile);
        unsigned int getPMVMU(TrainingMatrixTile* tile);
        unsigned int getPTile(TrainingMatrixTile* tile);
        unsigned int getPCore(TrainingMatrixTile* tile);
        unsigned int getPMVMU(Operation* op);
        unsigned int getPTile(Operation* op);
        unsigned int getPCore(Operation* op);

        std::string printAssignment(Operation* op);

};

