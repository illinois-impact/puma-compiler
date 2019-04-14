/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <map>
#include <set>
#include <vector>

#include "common.h"

class Coalescer {

    private:

        ModelImpl* model_;
        Placer* placer_;

        std::vector<std::set<MVMOperation*>*>& coalesceableMVMSets_;
        std::vector<std::vector<CoalescedMVMSet*>> coalescedMVMSets_;
        std::vector<std::vector<CoalescedTrainingOperationSet*>> coalescedTrainingOperationSets_;

        void coalesceMVMOperations();
        void findMVMPredecessors(Operation* op, std::map<Operation*, std::set<MVMOperation*>>& mvmPredecessors);
        void coalesceMVMPredecessors(Operation* op, std::set<Operation*>& isVisited, std::map<MVMOperation*, std::set<MVMOperation*>>& mvmPredecessorsOfMVMs, std::map<MVMOperation*, std::set<MVMOperation*>>& mvmSuccessorsOfMVMs);

        void coalesceTrainingOperations();
        void findImmediateTrainingOperationPredecessors(Operation* op, std::set<TrainingMatrixOperation*>& foundSet);
        void findAllTrainingOperationPredecessors(TrainingMatrixOperation* trainOp, std::set<TrainingMatrixOperation*>& foundSet, std::map<TrainingMatrixOperation*, std::set<TrainingMatrixOperation*>>& immediateTrainingOperationPredecessors);
        void coalesceTrainingOperationPredecessors(Operation* op, std::set<Operation*>& isVisited, std::map<TrainingMatrixOperation*, std::set<TrainingMatrixOperation*>>& trainingOperationPredecessors, std::map<TrainingMatrixOperation*, std::set<TrainingMatrixOperation*>>& trainingOperationSuccessors);

    public:

        Coalescer(ModelImpl* model, Placer* placer, std::vector<std::set<MVMOperation*>*>& coalesceableMVMSets);
        ~Coalescer();

};

