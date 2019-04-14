/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <map>

#include "common.h"

class ModelInstanceImpl {

    private:

        ModelImpl* model_;
        Placer* placer_;
        std::map<std::string, float*> tensorData_;

    public:

        ModelInstanceImpl(ModelImpl* model, Placer* placer);

        void bind(std::string tensorName, float* data);
        void generateData();

};

