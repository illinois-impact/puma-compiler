/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>
#include <algorithm>
#include <fstream>
#include <sstream>

#include "puma.h"

#include "model.h"
#include "operations.h"
#include "partitioner.h"
#include "tensors.h"

Partitioner::Partitioner(ModelImpl* model, CompilerOptions::GraphPartitioningScheme gp)
    : model_(model), gp_(gp)
{
    switch(gp_) {
        case CompilerOptions::GP_ROW_MAJOR:
            assignVMVMUsInRowMajor();
            assignVCoresInVMVMUOrder();
            assignVTilesInVMVMUOrder();
            break;
        case CompilerOptions::GP_COL_MAJOR:
            assignVMVMUsInColMajor();
            assignVCoresInVMVMUOrder();
            assignVTilesInVMVMUOrder();
            break;
        case CompilerOptions::GP_KAHIP: // FIXME: check whether KaHIP exists
            assignVMVMUsInRowMajor(); // Doesn't matter which order is used because KaHIP will partition agnostically
            assignVCoresWithKaHIP();
            assignVTilesWithKaHIP();
            break;
        case CompilerOptions::GP_RANDOM:
            assignVMVMUsRandomly();
            assignVCoresInVMVMUOrder();
            assignVTilesInVMVMUOrder();
            break;
        default: assert(0 && "Unrecognized graph partitioning scheme!");
    }
    insertLoadsAndStores();
    insertSendsAndRecives();
    insertInputAndOutput();
    insertCopies();
}

bool Partitioner::isVMVMUAssigned(Operation* op) {
    return op2vmvmu_.count(op);
}

void Partitioner::assignVMVMU(Operation* op, unsigned int vMVMU) {
    assert((!isVMVMUAssigned(op)) && "Cannot reassign virtual MVMU!");
    op2vmvmu_[op] = vMVMU;
}

void Partitioner::cloneAssignment(Operation* cloneFrom, Operation* cloneTo) {
    if(isVMVMUAssigned(cloneFrom)) {
        assignVMVMU(cloneTo, getVMVMU(cloneFrom));
    }
}

unsigned int Partitioner::getVMVMU(ConstantMatrixTile* tile) {
    assert(cmat2vmvmu_.count(tile) && "Virtual MVMU not assigned!");
    return cmat2vmvmu_[tile];
}

unsigned int Partitioner::getVCore(ConstantMatrixTile* tile) {
    return vmvmu2vcore_[getVMVMU(tile)];
}

unsigned int Partitioner::getVTile(ConstantMatrixTile* tile) {
    return vcore2vtile_[getVCore(tile)];
}

unsigned int Partitioner::getVMVMU(TrainingMatrixTile* tile) {
    assert(tmat2vmvmu_.count(tile) && "Virtual MVMU not assigned!");
    return tmat2vmvmu_[tile];
}

unsigned int Partitioner::getVCore(TrainingMatrixTile* tile) {
    return vmvmu2vcore_[getVMVMU(tile)];
}

unsigned int Partitioner::getVTile(TrainingMatrixTile* tile) {
    return vcore2vtile_[getVCore(tile)];
}

unsigned int Partitioner::getVMVMU(Operation* op) {
    assert(isVMVMUAssigned(op) && "Virtual MVMU not assigned!");
    return op2vmvmu_[op];
}

unsigned int Partitioner::getVCore(Operation* op) {
    return vmvmu2vcore_[getVMVMU(op)];
}

unsigned int Partitioner::getVTile(Operation* op) {
    return vcore2vtile_[getVCore(op)];
}

void Partitioner::assignVMVMUsInRowMajor() {

    // Extract all matrix tiles in row major order
    if(model_->getModelType() == ModelImpl::INFERENCE) {
        for(auto m = model_->const_mat_begin(); m != model_->const_mat_end(); ++m) {
            ConstantMatrixImpl* mat = *m;
            for(unsigned int h = 0; h < mat->nHeightTiles(); ++h) {
                for(unsigned int w = 0; w < mat->nWidthTiles(); ++w) {
                    cmatTiles_.push_back(mat->getTile(h, w));
                }
            }
        }
        for(auto m = model_->conv_mat_begin(); m != model_->conv_mat_end(); ++m) {
            ConvolutionalConstantMatrixImpl* mat = *m;
            for(unsigned int kh = 0; kh < mat->getKernelHeight(); ++kh) {
                for(unsigned int kw = 0; kw < mat->getKernelWidth(); ++kw) {
                    for(unsigned int h = 0; h < mat->getNOutChannelTiles(); ++h) {
                        for(unsigned int w = 0; w < mat->getNInChannelTiles(); ++w) {
                            cmatTiles_.push_back(mat->getTile(kh, kw, h, w));
                        }
                    }
                }
            }
        }
    } else if(model_->getModelType() == ModelImpl::TRAINING) {
        for(auto m = model_->train_mat_begin(); m != model_->train_mat_end(); ++m) {
            TrainingMatrixImpl* mat = *m;
            for(unsigned int h = 0; h < mat->nHeightTiles(); ++h) {
                for(unsigned int w = 0; w < mat->nWidthTiles(); ++w) {
                    tmatTiles_.push_back(mat->getTile(h, w));
                }
            }
        }
    }

    assignVMVMUsAndSpreadAffinity();

}

void Partitioner::assignVMVMUsInColMajor() {

    // Extract all matrix tiles in column major order
    if(model_->getModelType() == ModelImpl::INFERENCE) {
        for(auto m = model_->const_mat_begin(); m != model_->const_mat_end(); ++m) {
            ConstantMatrixImpl* mat = *m;
            for(unsigned int w = 0; w < mat->nWidthTiles(); ++w) {
                for(unsigned int h = 0; h < mat->nHeightTiles(); ++h) {
                    cmatTiles_.push_back(mat->getTile(h, w));
                }
            }
        }
        for(auto m = model_->conv_mat_begin(); m != model_->conv_mat_end(); ++m) {
            ConvolutionalConstantMatrixImpl* mat = *m;
            for(unsigned int kh = 0; kh < mat->getKernelHeight(); ++kh) {
                for(unsigned int kw = 0; kw < mat->getKernelWidth(); ++kw) {
                    for(unsigned int w = 0; w < mat->getNInChannelTiles(); ++w) {
                        for(unsigned int h = 0; h < mat->getNOutChannelTiles(); ++h) {
                            cmatTiles_.push_back(mat->getTile(kh, kw, h, w));
                        }
                    }
                }
            }
        }
    } else if(model_->getModelType() == ModelImpl::TRAINING) {
        for(auto m = model_->train_mat_begin(); m != model_->train_mat_end(); ++m) {
            TrainingMatrixImpl* mat = *m;
            for(unsigned int w = 0; w < mat->nWidthTiles(); ++w) {
                for(unsigned int h = 0; h < mat->nHeightTiles(); ++h) {
                    tmatTiles_.push_back(mat->getTile(h, w));
                }
            }
        }
    }

    assignVMVMUsAndSpreadAffinity();

}

void Partitioner::assignVMVMUsRandomly() {

    // Extract all matrix tiles in row major order
    if(model_->getModelType() == ModelImpl::INFERENCE) {
        for(auto m = model_->const_mat_begin(); m != model_->const_mat_end(); ++m) {
            ConstantMatrixImpl* mat = *m;
            for(unsigned int h = 0; h < mat->nHeightTiles(); ++h) {
                for(unsigned int w = 0; w < mat->nWidthTiles(); ++w) {
                    cmatTiles_.push_back(mat->getTile(h, w));
                }
            }
        }
        for(auto m = model_->conv_mat_begin(); m != model_->conv_mat_end(); ++m) {
            ConvolutionalConstantMatrixImpl* mat = *m;
            for(unsigned int kh = 0; kh < mat->getKernelHeight(); ++kh) {
                for(unsigned int kw = 0; kw < mat->getKernelWidth(); ++kw) {
                    for(unsigned int h = 0; h < mat->getNOutChannelTiles(); ++h) {
                        for(unsigned int w = 0; w < mat->getNInChannelTiles(); ++w) {
                            cmatTiles_.push_back(mat->getTile(kh, kw, h, w));
                        }
                    }
                }
            }
        }
    } else if(model_->getModelType() == ModelImpl::TRAINING) {
        for(auto m = model_->train_mat_begin(); m != model_->train_mat_end(); ++m) {
            TrainingMatrixImpl* mat = *m;
            for(unsigned int h = 0; h < mat->nHeightTiles(); ++h) {
                for(unsigned int w = 0; w < mat->nWidthTiles(); ++w) {
                    tmatTiles_.push_back(mat->getTile(h, w));
                }
            }
        }
    }

    // Shuffle them randomly
    if(model_->getModelType() == ModelImpl::INFERENCE) {
        std::random_shuffle(cmatTiles_.begin(), cmatTiles_.end());
    } else if(model_->getModelType() == ModelImpl::TRAINING) {
        std::random_shuffle(tmatTiles_.begin(), tmatTiles_.end());
    }

    assignVMVMUsAndSpreadAffinity();

}

void Partitioner::assignVMVMUsAndSpreadAffinity() {

    // Reserve virtual MVMUs 0 and 1 for input and output tiles respectively
    nVMVMUs_ = 2;

    // Assign matrix tiles to virtual MVMUs
    if(model_->getModelType() == ModelImpl::INFERENCE) {
        for(ConstantMatrixTile* tile : cmatTiles_) {
            unsigned int vMVMU = nVMVMUs_++;
            cmat2vmvmu_[tile] = vMVMU;
            for(unsigned int u = 0; u < tile->numUsers(); ++u) {
                MVMOperation* mvm = tile->getUser(u);
                assignVMVMU(mvm, vMVMU);
                spreadVMVMUAffinityToOperands(mvm);
                spreadVMVMUAffinityToUsers(mvm);
            }
        }
    } else if(model_->getModelType() == ModelImpl::TRAINING) {
        for(TrainingMatrixTile* tile : tmatTiles_) {
            unsigned int vMVMU = nVMVMUs_++;
            tmat2vmvmu_[tile] = vMVMU;
            for(unsigned int u = 0; u < tile->numUsers(); ++u) {
                TrainingMatrixOperation* trainOp = tile->getUser(u);
                assignVMVMU(trainOp, vMVMU);
                spreadVMVMUAffinityToOperands(trainOp);
                spreadVMVMUAffinityToUsers(trainOp);
            }
        }
    }

    // Resolve assignment for operations with operands from different virtual MVMUs
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(op)) {
            if(!isVMVMUAssigned(consumer)) {
                for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
                    ProducerOperation* operand = consumer->getOperand(o);
                    if(isVMVMUAssigned(operand)) {
                        // TODO: Heuristic for which MVMU to assign to if not all operands are assigned to MVMUs.
                        //       Currently assigning to MVMU of first operand that is assigned (if any).
                        cloneAssignment(operand, consumer);
                        spreadVMVMUAffinityToOperands(consumer);
                        if(ProducerOperation* producer = dynamic_cast<ProducerOperation*>(consumer)) {
                            spreadVMVMUAffinityToUsers(producer);
                        }
                        break;
                    }
                }
            }
        }
    }

}

void Partitioner::spreadVMVMUAffinityToOperands(ConsumerOperation* op) {
    for(unsigned int o = 0; o < op->numOperands(); ++o) {
        ProducerOperation* producer = op->getOperand(o);
        if(!isVMVMUAssigned(producer) && !dynamic_cast<MVMOperation*>(producer) && !dynamic_cast<TrainingMatrixOperation*>(producer)) {
            bool allUsersAssigned = true;
            for(auto u = producer->user_begin(); u != producer->user_end(); ++u) {
                ConsumerOperation* consumer = *u;
                if(!isVMVMUAssigned(consumer)) {
                    allUsersAssigned = false;
                    break;
                }
            }
            if(allUsersAssigned) {
                // TODO: Heuristic for which MVMU to select if users assigned to different MVMUs.
                //       Currently just assigning to same MVMU as last user processed.
                cloneAssignment(op, producer);
                if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(producer)) {
                    spreadVMVMUAffinityToOperands(consumer);
                }
            }
        }
    }
}

void Partitioner::spreadVMVMUAffinityToUsers(ProducerOperation* op) {
    for(auto u = op->user_begin(); u != op->user_end(); ++u) {
        ConsumerOperation* consumer = *u;
        if(!isVMVMUAssigned(consumer) && !dynamic_cast<MVMOperation*>(consumer) && !dynamic_cast<TrainingMatrixOperation*>(consumer)) {
            bool allOperandsAssigned = true;
            for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
                ProducerOperation* producer = consumer->getOperand(o);
                if(!isVMVMUAssigned(producer)) {
                    allOperandsAssigned = false;
                    break;
                }
            }
            if(allOperandsAssigned) {
                // TODO: Heuristic for which MVMU to select if operands assigned to different MVMUs.
                //       Currently just assigning to same MVMU as last operand processed.
                cloneAssignment(op, consumer);
                if(ProducerOperation* producer = dynamic_cast<ProducerOperation*>(consumer)) {
                    spreadVMVMUAffinityToUsers(producer);
                }
            }
        }
    }

}

void Partitioner::assignVCoresInVMVMUOrder() {

    vmvmu2vcore_.resize(nVMVMUs_);

    // Reserve virtual cores 0 and 1 for input and output tiles respectively
    nVCores_ = 2;
    vmvmu2vcore_[0] = 0;
    vmvmu2vcore_[1] = 1;

    // Assign virtual MVMUs to virtual cores in order
    unsigned int nMVMUSPerCore = (model_->getModelType() == ModelImpl::INFERENCE)?(N_CONSTANT_MVMUS_PER_CORE):(N_TRAINING_MVMUS_PER_CORE);
    nVCores_ += (nVMVMUs_ - 2 - 1)/nMVMUSPerCore + 1; // -2 accounts for virtual MVMUs 0 and 1 which are reserved for input and output
    for(unsigned int vMVMU = 2; vMVMU < nVMVMUs_; ++vMVMU) {
        vmvmu2vcore_[vMVMU] = (vMVMU - 2)/nMVMUSPerCore + 2;
    }

}

void Partitioner::assignVTilesInVMVMUOrder() {

    vcore2vtile_.resize(nVCores_);

    // Reserve virtual tiles 0 and 1 for input and output tiles respectively
    nVTiles_ = 2;
    vcore2vtile_[0] = 0;
    vcore2vtile_[1] = 1;

    // Assign virtual cores to virtual tiles in order
    nVTiles_ += (nVCores_ - 2 - 1)/N_CORES_PER_TILE + 1; // -2 accounts for virtual cores 0 and 1 which are reserved for input and output
    for(unsigned int vCore = 2; vCore < nVCores_; ++vCore) {
        vcore2vtile_[vCore] = (vCore - 2)/N_CORES_PER_TILE + 2;;
    }

}

void partitionGraphWithKaHIP(unsigned int numNodes, unsigned int numEdges, unsigned int numNodesPerPartition, std::vector<std::pair<unsigned int, unsigned int>>* edges, std::vector<unsigned int>& result) {

    // Output graph file
    std::ofstream graphOut("kahip_input.graph", std::ofstream::out);
    graphOut << numNodes << " " << numEdges << " 11" << std::endl;
    for(unsigned int node = 0; node < numNodes; ++node) {
        graphOut << "1 "; // All nodes have weight 1
        for(auto edge : edges[node]) {
            graphOut << edge.first + 1 /* destination node */ << " " << edge.second /* edge weight */ << " ";
        }
        graphOut << std::endl;
    }
    graphOut.close();

    // Execute KaHIP command
    std::stringstream cmd; 
    unsigned int numPartitions = (numNodes - 1)/numNodesPerPartition + 1;
    double imbalance = (double)(numPartitions*numNodesPerPartition)/((double)(numNodes)) - 1;
    cmd << "kaffpaE ./kahip_input.graph --k=" << numPartitions << " --imbalance="<< imbalance
        << " --preconfiguration=strong --output_filename=kahip_partition_result"; 
    system(cmd.str().c_str());

    // Input result file
    std::ifstream resultIn("kahip_partition_result", std::ifstream::in);
    for(unsigned int node = 0; node < numNodes; ++node) {
        resultIn >> result[node];
    }
    resultIn.close();

}

void Partitioner::assignVCoresWithKaHIP() {

    // Build graph
    unsigned int numNodes = nVMVMUs_ - 2;
    unsigned int numEdges = 0;
    std::vector<std::pair<unsigned int, unsigned int>> edges[numNodes];
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if(ProducerOperation* producer = dynamic_cast<ProducerOperation*>(op)) {
            unsigned int producerNodeID = getVMVMU(producer) - 2;
            for(auto u = producer->user_begin(); u != producer->user_end(); ++u) {
                ConsumerOperation* consumer = *u;
                unsigned int consumerNodeID = getVMVMU(consumer) - 2;
                if(producerNodeID != consumerNodeID) {
                    edges[producerNodeID].push_back(std::make_pair(consumerNodeID, producer->length()));
                    edges[consumerNodeID].push_back(std::make_pair(producerNodeID, producer->length()));
                    ++numEdges;
                }
            }
        }
    }

    // Call KaHIP
    unsigned int nMVMUSPerCore = (model_->getModelType() == ModelImpl::INFERENCE)?(N_CONSTANT_MVMUS_PER_CORE):(N_TRAINING_MVMUS_PER_CORE);
    unsigned int numNodesPerPartition = nMVMUSPerCore;
    std::vector<unsigned int> result(numNodes);
    partitionGraphWithKaHIP(numNodes, numEdges, numNodesPerPartition, edges, result);

    // Process result
    unsigned int numPartitions = (numNodes - 1)/numNodesPerPartition + 1;
    nVCores_ = numPartitions + 2;
    vmvmu2vcore_.resize(nVMVMUs_);
    vmvmu2vcore_[0] = 0;
    vmvmu2vcore_[1] = 1;
    for(unsigned int node = 0; node < numNodes; ++node) {
        vmvmu2vcore_[node + 2] = result[node] + 2;
    }

}

void Partitioner::assignVTilesWithKaHIP() {

    // Build graph
    unsigned int numNodes = nVCores_ - 2;
    unsigned int numEdges = 0;
    std::vector<std::pair<unsigned int, unsigned int>> edges[numNodes];
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if(ProducerOperation* producer = dynamic_cast<ProducerOperation*>(op)) {
            unsigned int producerNodeID = getVCore(producer) - 2;
            for(auto u = producer->user_begin(); u != producer->user_end(); ++u) {
                ConsumerOperation* consumer = *u;
                unsigned int consumerNodeID = getVCore(producer) - 2;
                if(producerNodeID != consumerNodeID) {
                    edges[producerNodeID].push_back(std::make_pair(consumerNodeID, producer->length()));
                    edges[consumerNodeID].push_back(std::make_pair(producerNodeID, producer->length()));
                    ++numEdges;
                }
            }
        }
    }

    // Call KaHIP
    unsigned int numNodesPerPartition = N_CORES_PER_TILE;
    std::vector<unsigned int> result(numNodes);
    partitionGraphWithKaHIP(numNodes, numEdges, numNodesPerPartition, edges, result);

    // Process result
    unsigned int numPartitions = (numNodes - 1)/numNodesPerPartition + 1;
    nVTiles_ = numPartitions + 2;
    vcore2vtile_.resize(nVCores_);
    vcore2vtile_[0] = 0;
    vcore2vtile_[1] = 1;
    for(unsigned int node = 0; node < numNodes; ++node) {
        vcore2vtile_[node + 2] = result[node] + 2;
    }

}

void Partitioner::insertLoadsAndStores() {

    // Insert loads and stores across cores
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if(ProducerOperation* producer = dynamic_cast<ProducerOperation*>(op)) {
            StoreOperation* store = NULL;
            std::map<unsigned int, LoadOperation*> loads;
            for(auto u = producer->user_begin(); u != producer->user_end(); ) {
                ConsumerOperation* consumer = *u;
                ++u; // replaceOperand may remove consumer from producer's users
                if(getVCore(producer) != getVCore(consumer)) {
                    if(store == NULL) {
                        store = new StoreOperation(model_, producer);
                        numStores_ += store->length();
                        cloneAssignment(producer, store);
                    }
                    if(loads[getVCore(consumer)] == NULL) {
                        LoadOperation* load = new LoadOperation(model_, store);
                        numLoads_ += load->length();
                        cloneAssignment(consumer, load);
                        loads[getVCore(consumer)] = load;
                    }
                    consumer->replaceOperand(producer, loads[getVCore(consumer)]);
                }
            }
        }
    }

}

void Partitioner::insertSendsAndRecives() {

    // Insert sends and receives across tiles
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if(StoreOperation* store = dynamic_cast<StoreOperation*>(op)) {
            std::map<unsigned int, ReceiveOperation*> recvs;
            for(auto u = store->user_begin(); u != store->user_end(); ) {
                TileMemoryReadOperation* read = *u;
                ++u; // replaceSrc may remove read from store's users
                if(getVTile(store) != getVTile(read)) {
                    if(recvs[getVTile(read)] == NULL) {
                        SendOperation* send = new SendOperation(model_, store);
                        numSends_ += send->length();
                        cloneAssignment(store, send);
                        ReceiveOperation* recv = new ReceiveOperation(model_, send);
                        numReceives_ += recv->length();
                        cloneAssignment(read, recv);
                        recvs[getVTile(read)] = recv;
                    }
                    read->replaceSrc(store, recvs[getVTile(read)]);
                }
            }
        }
    }

}

void Partitioner::insertInputAndOutput() {

    // Replace pseudo input and output operations
    std::map<InputVectorTile*, std::map<unsigned int, LoadOperation*>> loads;
    std::map<InputVectorTile*, std::map<unsigned int, ReceiveOperation*>> recvs;
    std::map<InputVectorTile*, WriteInputOperation*> inputs;
    for(auto it = model_->op_begin(); it != model_->op_end(); ) {
        Operation* op = *it;
        ++it; // op might get removed from the graph
        if(PseudoInputOperation* pseudoInput = dynamic_cast<PseudoInputOperation*>(op)) {
            InputVectorTile* src = pseudoInput->getSrc();
            for(auto u = pseudoInput->user_begin(); u != pseudoInput->user_end(); ) {
                ConsumerOperation* consumer = *u;
                ++u; // replaceOperand may remove consumer from pseudoInput's users
                if(loads[src][getVCore(consumer)] == NULL) {
                    if(recvs[src][getVTile(consumer)] == NULL) {
                        if(inputs[src] == NULL) {
                            WriteInputOperation* input = new WriteInputOperation(model_, src);
                            assignVMVMU(input, 0);
                            inputs[src] = input;
                        }
                        SendOperation* send = new SendOperation(model_, inputs[src]);
                        numSends_ += send->length();
                        cloneAssignment(inputs[src], send);
                        ReceiveOperation* recv = new ReceiveOperation(model_, send);
                        numReceives_ += recv->length();
                        cloneAssignment(consumer, recv);
                        recvs[src][getVTile(consumer)] = recv;
                    }
                    LoadOperation* load = new LoadOperation(model_, recvs[src][getVTile(consumer)]);
                    numLoads_ += load->length();
                    cloneAssignment(consumer, load);
                    loads[src][getVCore(consumer)] = load;
                }
                consumer->replaceOperand(pseudoInput, loads[src][getVCore(consumer)]);
            }
            unlink(pseudoInput);
        } else if(PseudoOutputOperation* pseudoOutput = dynamic_cast<PseudoOutputOperation*>(op)) {
            OutputVectorTile* dst = pseudoOutput->getDst();
            for(unsigned int o = 0; o < pseudoOutput->numOperands(); ++o) {
                ProducerOperation* producer = pseudoOutput->getOperand(o);
                StoreOperation* store = new StoreOperation(model_, producer);
                numStores_ += store->length();
                cloneAssignment(pseudoOutput, store);
                SendOperation* send = new SendOperation(model_, store);
                numSends_ += send->length();
                cloneAssignment(pseudoOutput, send);
                ReceiveOperation* recv = new ReceiveOperation(model_, send);
                numReceives_ += recv->length();
                assignVMVMU(recv, 1);
                ReadOutputOperation* output = new ReadOutputOperation(model_, recv, dst);
                cloneAssignment(recv, output);
                producer->removeUser(pseudoOutput);
            }
            unlink(pseudoOutput);
        }
    }

}

void Partitioner::insertCopies() {

    // Insert copy operations across producers and consumers that use different register spaces
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(op)) {
            bool isMatrixOperation = (dynamic_cast<MVMOperation*>(consumer) != NULL) || (dynamic_cast<TrainingMatrixOperation*>(consumer) != NULL);
            if(isMatrixOperation) {
                for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
                    ProducerOperation* producer = consumer->getOperand(o);
                    bool producerIsMatrixOperation = (dynamic_cast<MVMOperation*>(consumer) != NULL) || (dynamic_cast<TrainingMatrixOperation*>(consumer) != NULL);
                    bool producerHasMultipleUsers = (producer->numUsers() > 1);
                    if(producerIsMatrixOperation || producerHasMultipleUsers) {
                        /*
                         * producerIsMatrixOperation:
                         * Matrix operations write their outputs to reserved output registers which cannot be read by other matrix
                         * operations which read from reserved input registers. Therefore if a matrix operation feeds another matrix
                         * operation, a copy must be inserted between them.
                         *
                         * producerHasMultipleUsers:
                         * If a producer has multiple users, it can't be assigned to the reserved input registers of one matrix
                         * operation because the other users also need to access it. Therefore, a copy is inserted to the matix
                         * operation.
                         */
                        CopyOperation* copy = new CopyOperation(model_, producer);
                        cloneAssignment(consumer, copy);
                        consumer->replaceOperand(producer, copy);
                    }
                }
            }
        }
    }

}

void Partitioner::unlink(Operation* op) {
    op2vmvmu_.erase(op);
    model_->unlink(op);
}

std::string Partitioner::printAssignment(Operation* op) {
    std::stringstream ss;
    if(isVMVMUAssigned(op)) {
        ss << "\nvMVMU = " << getVMVMU(op);
    }
    if(vmvmu2vcore_.size() > 0) {
        ss << ", vCore = " << getVCore(op);
    }
    if(vcore2vtile_.size() > 0) {
        ss << ", vTile = " << getVTile(op);
    }
    return ss.str();
}

void Partitioner::printReport(std::ofstream& report) {
    switch(gp_) {
        case CompilerOptions::GP_ROW_MAJOR:
            report << "graph partitioning scheme = row major" << std::endl;
            break;
        case CompilerOptions::GP_COL_MAJOR:
            report << "graph partitioning scheme = column major" << std::endl;
            break;
        case CompilerOptions::GP_KAHIP: // FIXME: check whether KaHIP exists
            report << "graph partitioning scheme = KaHIP" << std::endl;
            break;
        case CompilerOptions::GP_RANDOM:
            report << "graph partitioning scheme = random" << std::endl;
            break;
        default: assert(0 && "Unrecognized graph partitioning scheme!");
    }
    report << "# load bytes = " << numLoads_ << std::endl;
    report << "# store bytes = " << numStores_ << std::endl;
    report << "# load + store bytes = " << numLoads_ + numStores_ << std::endl;
    report << "# send bytes = " << numSends_ << std::endl;
    report << "# receive bytes = " << numReceives_ << std::endl;
    report << "# send + receive bytes = " << numSends_ + numReceives_ << std::endl;
}

