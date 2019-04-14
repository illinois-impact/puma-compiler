/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <list>
#include <set>
#include <vector>

#include "common.h"

class Linearizer {

    private:

        ModelImpl* model_;
        Partitioner* partitioner_;
        Placer* placer_;

        std::vector<std::list<CoreOperation*>> coreOperationLists_;
        std::vector<std::list<TileOperation*>> tileOperationLists_;

        void linearize();
        void linearizeWithPredecessors(Operation* op, std::set<Operation*>& isVisited, std::set<Operation*>& wasAddedEarly, bool addSelf=true);
        void addToList(Operation* op, std::set<Operation*>& isVisited);
        void addConsumersToList(ProducerOperation* producer, std::set<Operation*>& isVisited, std::set<Operation*>& wasAddedEarly);

    public:

        Linearizer(ModelImpl* model, Partitioner* partitioner, Placer* placer);

        std::list<CoreOperation*>& getCoreOperationList(unsigned int pTile, unsigned int pCore);
        std::list<TileOperation*>& getTileOperationList(unsigned int pTile);

};

