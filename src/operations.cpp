/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>
#include <sstream>

#include "model.h"
#include "operations.h"
#include "tensors.h"

void OutputVector::operator=(Vector xparam) {
    VectorImpl* x = xparam.unwrap();
    OutputVectorImpl* y = impl_;
    y->checkCompatibility(x);
    for(unsigned int t = 0; t < x->nTiles(); ++t) {
        ProducerOperation* producer = x->getTile(t);
        OutputVectorTile* output = y->getTile(t);
        new PseudoOutputOperation(producer->getModel(), producer, output);
    }
}

void OutputImagePixelStream::operator=(ImagePixelStream xsparam) {
    ImagePixelStreamImpl* xs = xsparam.unwrap();
    OutputImagePixelStreamImpl* ys = impl_;
    ys->checkCompatibility(xs);
    for(unsigned int t = 0; t < xs->nTiles(); ++t) {
        ImagePixelStreamTile* xsTile = xs->getTile(t);
        OutputImagePixelStreamTile* ysTile = ys->getTile(t);
        // TODO: Convert the following into a single operation with codegened loops
        for(unsigned int h = 0; h < xs->imageHeight(); ++h) {
            for(unsigned int w = 0; w < xs->imageWidth(); ++w) {
                ProducerOperation* x = xsTile->get(h, w);
                OutputVectorTile* y = ysTile->get(h, w);
                new PseudoOutputOperation(x->getModel(), x, y);
            }
        }
    }
}

Vector::Vector(InputVector xparam) {
    InputVectorImpl* x = xparam.unwrap();
    VectorImpl* y = new VectorImpl(x->getModel(), x->length());
    y->checkCompatibility(x);
    for(unsigned int t = 0; t < x->nTiles(); ++t) {
        ProducerOperation* producer = new PseudoInputOperation(x->getModel(), x->getTile(t));
        y->setTile(t, producer);
    }
    impl_ = y;
}

ImagePixelStream::ImagePixelStream(InputImagePixelStream xsparam) {
    InputImagePixelStreamImpl* xs = xsparam.unwrap();
    ImagePixelStreamImpl* ys = new ImagePixelStreamImpl(xs->getModel(), xs->imageWidth(), xs->imageHeight(), xs->nChannels());
    ys->checkCompatibility(xs);
    for(unsigned int t = 0; t < xs->nTiles(); ++t) {
        InputImagePixelStreamTile* xsTile = xs->getTile(t);
        ImagePixelStreamTile* ysTile = ys->getTile(t);
        // TODO: Convert the following into a single operation with codegened loops
        for(unsigned int h = 0; h < xs->imageHeight(); ++ h) {
            for(unsigned int w = 0; w < xs->imageWidth(); ++ w) {
                InputVectorTile* x = xsTile->get(h, w);
                ProducerOperation* y = new PseudoInputOperation(x->getModel(), x);
                ysTile->add(h, w, y);
            }
        }
    }
    impl_ = ys;
}

Vector unaryOp(Vector xparam, ALUVectorOperation::OpCode op) {
    VectorImpl* x = xparam.unwrap();
    VectorImpl* y = new VectorImpl(x->getModel(), x->length());
    y->checkCompatibility(x);
    for(unsigned int t = 0; t < x->nTiles(); ++t) {
        ProducerOperation* producer = new ALUVectorOperation(x->getModel(), op, x->getTile(t));
        y->setTile(t, producer);
    }
    return Vector(y);
}

Vector operator~(Vector x) {
    return unaryOp(x, ALUVectorOperation::NOT);
}

Vector sig(Vector x) {
    return unaryOp(x, ALUVectorOperation::SIG);
}

Vector tanh(Vector x) {
    return unaryOp(x, ALUVectorOperation::TANH);
}

Vector exp(Vector x) {
    return unaryOp(x, ALUVectorOperation::EXP);
}

Vector log(Vector x) {
    return unaryOp(x, ALUVectorOperation::LOG);
}

Vector relu(Vector x) {
    return unaryOp(x, ALUVectorOperation::RELU);
}

Vector relud(Vector x) {
    return unaryOp(x, ALUVectorOperation::RELUD);
}

Vector log_softmax(Vector x) {
    return unaryOp(x, ALUVectorOperation::LOG_SOFTMAX);
}

Vector log_softmaxd(Vector x) {
    return unaryOp(x, ALUVectorOperation::LOG_SOFTMAXD);
}

Vector rndcmp(Vector x) {
    return unaryOp(x, ALUVectorOperation::RNDCMP);
}

Vector binaryOp(Vector x1param, Vector x2param, ALUVectorOperation::OpCode op) {
    VectorImpl* x1 = x1param.unwrap();
    VectorImpl* x2 = x2param.unwrap();
    VectorImpl* y = new VectorImpl(x1->getModel(), x1->length());
    y->checkCompatibility(x1);
    y->checkCompatibility(x2);
    for(unsigned int t = 0; t < x1->nTiles(); ++t) {
        ProducerOperation* producer = new ALUVectorOperation(x1->getModel(), op, x1->getTile(t), x2->getTile(t));
        y->setTile(t, producer);
    }
    return Vector(y);
}

Vector operator+(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::ADD);
}

Vector operator-(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::SUB);
}

Vector operator*(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::MUL);
}

Vector operator/(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::DIV);
}

Vector operator&(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::AND);
}

Vector operator|(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::OR);
}

Vector operator==(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::EQ);
}

Vector operator!=(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::NEQ);
}

Vector operator<(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::LT);
}

Vector operator<=(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::LEQ);
}

Vector operator>(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::GT);
}

Vector operator>=(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::GEQ);
}

Vector min(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::MIN);
}

Vector max(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::MAX);
}

Vector mse(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::MSE);
}

Vector immediateOp(Vector xparam, float imm, ALUVectorOperation::OpCode op) {
    VectorImpl* x = xparam.unwrap();
    VectorImpl* y = new VectorImpl(x->getModel(), x->length());
    y->checkCompatibility(x);
    for(unsigned int t = 0; t < x->nTiles(); ++t) {
        ProducerOperation* producer = new ALUVectorOperation(x->getModel(), op, x->getTile(t), imm);
        y->setTile(t, producer);
    }
    return Vector(y);
}

Vector operator*(float imm, Vector x) {
    return immediateOp(x, imm, ALUVectorOperation::MULI);
}

ImagePixelStream sig(ImagePixelStream xsparam) {
    ImagePixelStreamImpl* xs = xsparam.unwrap();
    ImagePixelStreamImpl* ys = new ImagePixelStreamImpl(xs->getModel(), xs->imageWidth(), xs->imageHeight(), xs->nChannels());
    ys->checkCompatibility(xs);
    for(unsigned int t = 0; t < xs->nTiles(); ++t) {
        ImagePixelStreamTile* xsTile = xs->getTile(t);
        ImagePixelStreamTile* ysTile = ys->getTile(t);
        // TODO: Convert the following into a single operation with codegened loops
        for(unsigned int h = 0; h < xs->imageHeight(); ++h) {
            for(unsigned int w = 0; w < xs->imageWidth(); ++w) {
                ProducerOperation* x = xsTile->get(h, w);
                ProducerOperation* y = new ALUVectorOperation(x->getModel(), ALUVectorOperation::SIG, x);
                ysTile->add(h, w, y);
            }
        }
    }
    return ImagePixelStream(ys);
}

ImagePixelStream maxpool(ImagePixelStream xsparam, unsigned int hspan, unsigned int wspan) {
    ImagePixelStreamImpl* xs = xsparam.unwrap();
    unsigned int ysWidth = (xs->imageWidth() - 1)/wspan + 1;
    unsigned int ysHeight = (xs->imageHeight() - 1)/hspan + 1;
    ImagePixelStreamImpl* ys = new ImagePixelStreamImpl(xs->getModel(), ysWidth, ysHeight, xs->nChannels());
    for(unsigned int t = 0; t < xs->nTiles(); ++t) {
        ImagePixelStreamTile* xsTile = xs->getTile(t);
        ImagePixelStreamTile* ysTile = ys->getTile(t);
        // TODO: Convert the following into a single operation with codegened loops
        ProducerOperation* accum[ysHeight][ysWidth][hspan*wspan];
        for(unsigned int hi = 0; hi < xs->imageHeight(); ++hi) {
            for(unsigned int wi = 0; wi < xs->imageWidth(); ++wi) {
                ProducerOperation* xTile = xsTile->get(hi, wi);
                unsigned int ho = hi/hspan;
                unsigned int hh = hi%hspan;
                unsigned int wo = wi/wspan;
                unsigned int ww = wi%wspan;
                unsigned int accumIdx = hh*wspan + ww;
                if(accumIdx == 0) {
                    accum[ho][wo][accumIdx] = xTile;
                } else {
                    accum[ho][wo][accumIdx] = new ALUVectorOperation(accum[ho][wo][accumIdx - 1]->getModel(), ALUVectorOperation::MAX, accum[ho][wo][accumIdx - 1], xTile);
                }
                if((hh == hspan - 1 || hi == xs->imageHeight() - 1) && (ww == wspan - 1 || wi == xs->imageWidth() - 1)) {
                    ysTile->add(ho, wo, accum[ho][wo][accumIdx]);
                }
            }
        }
    }
    return ImagePixelStream(ys);
}

Vector operator*(ConstantMatrix Mparam, Vector xparam) {
    ConstantMatrixImpl* M = Mparam.unwrap();
    ModelImpl* model = M->getModel();
    VectorImpl* x = xparam.unwrap();
    VectorImpl* y = new VectorImpl(model, M->height());
    M->checkCompatibilityForMVM(x);
    std::set<MVMOperation*>* coalesceableMVMSet = new std::set<MVMOperation*>();
    for(unsigned int h = 0; h < y->nTiles(); ++h) {
        ProducerOperation* accum[x->nTiles()]; // TODO: The following implements a sequential reduction; a tree reduction would be more efficient
        for(unsigned int w = 0; w < x->nTiles(); ++w) {
            MVMOperation* mvm = new MVMOperation(model, M->getTile(h, w), x->getTile(w));
            coalesceableMVMSet->insert(mvm);
            if(w == 0) {
                accum[w] = mvm;
            } else {
                accum[w] = new ALUVectorOperation(model, ALUVectorOperation::ADD, mvm, accum[w - 1]);
            }
        }
        y->setTile(h, accum[x->nTiles() - 1]);
    }
    model->addCoalesceableMVMSet(coalesceableMVMSet);
    return Vector(y);
}

ImagePixelStream operator*(ConvolutionalConstantMatrix Mparam, ImagePixelStream xsparam) {
    ConvolutionalConstantMatrixImpl* M = Mparam.unwrap();
    ImagePixelStreamImpl* xs = xsparam.unwrap();
    M->checkCompatibility(xs);
    ModelImpl* model = M->getModel();
    int kernelWidth = M->getKernelWidth();
    int kernelHeight = M->getKernelHeight();
    int nInChannelTiles = M->getNInChannelTiles();
    int imageWidth = xs->imageWidth();
    int imageHeight = xs->imageHeight();
    ImagePixelStreamImpl* ys[kernelHeight*kernelWidth*nInChannelTiles];
    for(int kh = 0; kh < kernelHeight; ++kh) { // Instantiates tiles within the same accumulation
        for(int kw = 0; kw < kernelWidth; ++kw) { // Instantiates tiles within the same accumulation
            for(int w = 0; w < nInChannelTiles; ++w) { // Instantiates tiles within the same accumulation
                int accumIdx = (kh*kernelWidth + kw)*nInChannelTiles + w;
                ys[accumIdx] = new ImagePixelStreamImpl(model, imageWidth, imageHeight, M->getNOutChannels());
                for(int h = 0; h < M->getNOutChannelTiles(); ++h) { // Instantiates independent tiles
                    ConstantMatrixTile* mat = M->getTile(kh, kw, h, w);
                    ImagePixelStreamTile* imageStream = xs->getTile(w);
                    ImagePixelStreamTile* accumStreamIn = (accumIdx == 0)?NULL:ys[accumIdx - 1]->getTile(h); // Partial sum feeding in from previous tile in the same accumulation
                    ImagePixelStreamTile* accumStreamOut = ys[accumIdx]->getTile(h); // Partial sum feeding out to the next tile in the same accumulation
                    // TODO: Convert the following into a single operation with codegened loops
                    for(int hi = -kernelHeight/2; hi < imageHeight + kernelHeight/2; ++hi) { // Loops over padded pixels of streamed input image
                        for(int wi = -kernelHeight/2; wi < imageWidth + kernelHeight/2; ++wi) { // Loops over padded pixels of streamed input image
                            int ho = hi + kernelHeight/2 - kh;
                            int wo = wi + kernelWidth/2 - kw;
                            bool inputInBounds = hi >= 0
                                                && hi < imageHeight
                                                && wi >= 0
                                                && wi < imageWidth;
                            bool outputInBounds = ho >= 0
                                                && ho < imageHeight
                                                && wo >= 0
                                                && wo < imageWidth;
                            ProducerOperation* pixel = NULL;
                            if(inputInBounds) {
                                pixel = imageStream->get(hi, wi);
                            }
                            if(outputInBounds) {
                                ProducerOperation* producer;
                                if(inputInBounds) {
                                    producer = new MVMOperation(model, mat, pixel);
                                } else {
                                    producer = new SetImmediateOperation(model, 0, mat->height()); // Use 0 for input padding
                                }
                                // TODO: The following implements a sequential reduction; it would be more efficient to implement a tree reduction
                                if(accumIdx == 0) {
                                    accumStreamOut->add(ho, wo, producer);
                                } else {
                                    accumStreamOut->add(ho, wo, new ALUVectorOperation(model, ALUVectorOperation::ADD, producer, accumStreamIn->get(ho, wo)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return ImagePixelStream(ys[kernelHeight*kernelWidth*nInChannelTiles - 1]);
}

Vector operator*(TrainingMatrix Mparam, Vector xparam) {
    TrainingMatrixImpl* M = Mparam.unwrap();
    ModelImpl* model = M->getModel();
    VectorImpl* x = xparam.unwrap();
    VectorImpl* y = new VectorImpl(model, M->height());
    M->checkCompatibilityForMVM(x);
    // TODO: Track coalesceable operations
    for(unsigned int h = 0; h < y->nTiles(); ++h) {
        ProducerOperation* accum[x->nTiles()]; // TODO: The following implements a sequential reduction; a tree reduction would be more efficient
        for(unsigned int w = 0; w < x->nTiles(); ++w) {
            TrainingMatrixOperation* trainingOp = new TrainingMatrixOperation(model, M->getTile(h, w), TrainingMatrixOperation::MVM, x->getTile(w));
            if(w == 0) {
                accum[w] = trainingOp;
            } else {
                accum[w] = new ALUVectorOperation(model, ALUVectorOperation::ADD, trainingOp, accum[w - 1]);
            }
        }
        y->setTile(h, accum[x->nTiles() - 1]);
    }
    return Vector(y);
}

Vector operator*(Transpose Mparam, Vector xparam) {
    TrainingMatrixImpl* M = Mparam.unwrap();
    ModelImpl* model = M->getModel();
    VectorImpl* x = xparam.unwrap();
    VectorImpl* y = new VectorImpl(model, M->width());
    M->checkCompatibilityForMVMTranspose(x);
    // TODO: Track coalesceable operations
    for(unsigned int h = 0; h < y->nTiles(); ++h) {
        ProducerOperation* accum[x->nTiles()]; // TODO: The following implements a sequential reduction; a tree reduction would be more efficient
        for(unsigned int w = 0; w < x->nTiles(); ++w) {
            TrainingMatrixOperation* trainingOp = new TrainingMatrixOperation(model, M->getTile(w, h), TrainingMatrixOperation::MVM_TRANSPOSE, x->getTile(w));
            if(w == 0) {
                accum[w] = trainingOp;
            } else {
                accum[w] = new ALUVectorOperation(model, ALUVectorOperation::ADD, trainingOp, accum[w - 1]);
            }
        }
        y->setTile(h, accum[x->nTiles() - 1]);
    }
    return Vector(y);
}

void operator-=(TrainingMatrix Mparam, OuterProduct op) {
    TrainingMatrixImpl* M = Mparam.unwrap();
    ModelImpl* model = M->getModel();
    VectorImpl* x1 = op.unwrap1();
    VectorImpl* x2 = op.unwrap2();
    M->checkCompatibilityForOuterProductAccumulate(x1, x2);
    // TODO: Track coalesceable operations
    for(unsigned int h = 0; h < M->nHeightTiles(); ++h) {
        for(unsigned int w = 0; w < M->nWidthTiles(); ++w) {
            TrainingMatrixOperation* trainingOp = new TrainingMatrixOperation(model, M->getTile(h, w), TrainingMatrixOperation::OUTER_PRODUCT, x1->getTile(h), x2->getTile(w));
        }
    }
}

Operation::Operation(ModelImpl* model, unsigned int length) : model_(model), length_(length) {
    assert(model != NULL);
    model->addOperation(this);
}

ConsumerOperation::ConsumerOperation(ProducerOperation* op1, ProducerOperation* op2) {
    if(op1 != NULL) {
        operands_.push_back(op1);
        op1->addUser(this);
        if(op2 != NULL) {
            operands_.push_back(op2);
            op2->addUser(this);
        }
    } else {
        assert(op2 == NULL);
    }
}

TileMemoryReadOperation::TileMemoryReadOperation(TileMemoryWriteOperation* src1, TileMemoryWriteOperation* src2) {
    assert(src1 != NULL);
    srcs_.push_back(src1);
    src1->addUser(this);
    if(src2 != NULL) {
        srcs_.push_back(src2);
        src2->addUser(this);
    }
}

InputOperation::InputOperation(InputVectorTile* src) : src_(src) {
    assert(src != NULL);
}

OutputOperation::OutputOperation(OutputVectorTile* dst) : dst_(dst) {
    assert(dst != NULL);
}

MVMOperation::MVMOperation(ModelImpl* model, ConstantMatrixTile* mat, ProducerOperation* op) : Operation(model, mat->height()), ConsumerOperation(op), mat_(mat), coalescedSet_(NULL) {
    assert(mat != NULL && op != NULL && mat->width() == op->length());
    assert(mat->width() <= MVMU_DIM && mat->height() <= MVMU_DIM && "MVM operations larger than one MVMU are not supported");
    mat->addUser(this);
}

TrainingMatrixOperation::TrainingMatrixOperation(ModelImpl* model, TrainingMatrixTile* mat, OpType opType, ProducerOperation* src1, ProducerOperation* src2) : Operation(model, (opType != MVM_TRANSPOSE)?(mat->height()):(mat->width())), ConsumerOperation(src1, src2), mat_(mat), opType_(opType), coalescedSet_(NULL) {
    assert(mat != NULL && src1 != NULL);
    assert(mat->width() <= MVMU_DIM && mat->height() <= MVMU_DIM && "MVM operations larger than one MVMU are not supported");
    if(opType == MVM) {
        assert(mat->width() == src1->length());
        assert(src2 == NULL);
    } else if(opType == MVM_TRANSPOSE) {
        assert(mat->height() == src1->length());
        assert(src2 == NULL);
    } else if(opType == OUTER_PRODUCT) {
        assert(mat->height() == src1->length());
        assert(src2 != NULL && mat->width() == src2->length());
    } else {
        assert(0 && "Invalid operation type!");
    }
    mat->addUser(this);
}

ALUVectorOperation::ALUVectorOperation(ModelImpl* model, OpCode opCode, ProducerOperation* src1, ProducerOperation* src2) : Operation(model, src1->length()), ConsumerOperation(src1, src2), opCode_(opCode), imm_(0.0f) {
    assert(!isImmediate());
    assert(src1 != NULL);
    switch(opCode_) {
        case ADD:
        case SUB:
        case MUL:
        case DIV:
        case AND:
        case OR:
        case EQ:
        case NEQ:
        case LT:
        case LEQ:
        case GT:
        case GEQ:
        case MIN:
        case MAX:
        case MSE:
            assert(src2 != NULL && src1->length() == src2->length());
    }
}

ALUVectorOperation::ALUVectorOperation(ModelImpl* model, OpCode opCode, ProducerOperation* src1, float imm) : Operation(model, src1->length()), ConsumerOperation(src1), opCode_(opCode), imm_(imm) {
    assert(isImmediate());
    assert(src1 != NULL);
}

SetImmediateOperation::SetImmediateOperation(ModelImpl* model, unsigned int imm, unsigned int length) : Operation(model, length), imm_(imm) {
}

CopyOperation::CopyOperation(ModelImpl* model, ProducerOperation* src) : Operation(model, src->length()), ConsumerOperation(src) {
    assert(src != NULL);
}

LoadOperation::LoadOperation(ModelImpl* model, TileMemoryWriteOperation* src) : Operation(model, src->length()), TileMemoryReadOperation(src) {
}

StoreOperation::StoreOperation(ModelImpl* model, ProducerOperation* src) : Operation(model, src->length()), ConsumerOperation(src) {
    assert(src != NULL);
}

SendOperation::SendOperation(ModelImpl* model, TileMemoryWriteOperation* src) : Operation(model, src->length()), TileMemoryReadOperation(src), dst_(NULL) {
}

ReceiveOperation::ReceiveOperation(ModelImpl* model, SendOperation* src) : Operation(model, src->length()), src_(src) {
    src->setDst(this);
}

WriteInputOperation::WriteInputOperation(ModelImpl* model, InputVectorTile* src) : Operation(model, src->length()), InputOperation(src) {
}

ReadOutputOperation::ReadOutputOperation(ModelImpl* model, TileMemoryWriteOperation* src, OutputVectorTile* dst) : Operation(model, src->length()), TileMemoryReadOperation(src), OutputOperation(dst) {
    assert(src->length() == dst->length());
}

PseudoInputOperation::PseudoInputOperation(ModelImpl* model, InputVectorTile* src) : Operation(model, src->length()), InputOperation(src) {
}

PseudoOutputOperation::PseudoOutputOperation(ModelImpl* model, ProducerOperation* op, OutputVectorTile* dst) : Operation(model, op->length()), ConsumerOperation(op), OutputOperation(dst) {
    assert(op != NULL && op->length() == dst->length());
}

void LoadOperation::addTileMemoryAddressOperand(ProducerOperation* address) {
    assert(operands_.size() == 0 && "Cannot set tile memory address operand!");
    assert(address->length() == 1 && "Address must be of length 1!");
    operands_.push_back(address);
    address->addUser(this);
}

void StoreOperation::addTileMemoryAddressOperand(ProducerOperation* address) {
    assert(operands_.size() == 1 && "Cannot set tile memory address operand!");
    assert(address->length() == 1 && "Address must be of length 1!");
    operands_.push_back(address);
    address->addUser(this);
}

void SendOperation::setDst(ReceiveOperation* dst) {
    assert(dst_ == NULL && "Cannot reset destination of send operation");
    dst_ = dst;
}

bool ConsumerOperation::uses(ProducerOperation* op) {
    for(unsigned int i = 0; i < operands_.size(); ++i) {
        if(operands_[i] == op) {
            return true;
        }
    }
    return false;
}

void ConsumerOperation::replaceOperand(ProducerOperation* op, ProducerOperation* replacement) {
    for(unsigned int i = 0; i < operands_.size(); ++i) {
        if(operands_[i] == op) {
            operands_[i] = replacement;
            op->removeUser(this);
            replacement->addUser(this);
        }
    }
}

void TileMemoryReadOperation::replaceSrc(TileMemoryWriteOperation* old, TileMemoryWriteOperation* replacement) {
    for(unsigned int i = 0; i < srcs_.size(); ++i) {
        if(srcs_[i] == old) {
            srcs_[i] = replacement;
            old->removeUser(this);
            replacement->addUser(this);
            return;
        }
    }
    assert(0 && "Source to be replaced not found");
}

void MVMOperation::setCoalescedSet(CoalescedMVMSet* coalescedSet) {
    assert(coalescedSet_ == NULL && "Cannot reassign coalesced set");
    coalescedSet_ = coalescedSet;
}

void MVMOperation::resetCoalescedSet() {
    coalescedSet_ = NULL;
}

void CoalescedMVMSet::add(MVMOperation* mvm, unsigned int pMVMU) {
    assert(mvms_[pMVMU] == NULL);
    mvms_[pMVMU] = mvm;
    mvm->setCoalescedSet(this);
}

void CoalescedMVMSet::removeAll() {
    for(unsigned int i = 0; i < mvms_.size(); ++i) {
        MVMOperation* mvm = mvms_[i];
        if(mvm != NULL) {
            mvms_[i] = NULL;
            mvm->resetCoalescedSet();
        }
    }
}

bool CoalescedMVMSet::isSetLeader(MVMOperation* mvm) {
    for(unsigned int i = 0; i < mvms_.size(); ++i) {
        MVMOperation* m = mvms_[i];
        if(m != NULL) {
            return (m == mvm); // Leader is first non-NULL MVM in the set
        }
    }
    assert(0 && "Unreachable: cannot have caolesced set with all NULL mvms!");
}

bool CoalescedMVMSet::isComplete() {
    for(auto mvm : mvms_) {
        if(mvm == NULL) {
            return false;
        }
    }
    return true;
}

void TrainingMatrixOperation::setCoalescedSet(CoalescedTrainingOperationSet* coalescedSet) {
    assert(coalescedSet_ == NULL && "Cannot reassign coalesced set");
    coalescedSet_ = coalescedSet;
}

void TrainingMatrixOperation::resetCoalescedSet() {
    coalescedSet_ = NULL;
}

void CoalescedTrainingOperationSet::add(TrainingMatrixOperation* trainOp, unsigned int pMVMU) {
    unsigned int index = pMVMU*N_TRAINING_OPERATIONS + trainOp->getOpType();
    assert(trainOps_[index] == NULL);
    trainOps_[index] = trainOp;
    trainOp->setCoalescedSet(this);
}

void CoalescedTrainingOperationSet::removeAll() {
    for(unsigned int i = 0; i < trainOps_.size(); ++i) {
        TrainingMatrixOperation* trainOp = trainOps_[i];
        if(trainOp != NULL) {
            trainOps_[i] = NULL;
            trainOp->resetCoalescedSet();
        }
    }
}

bool CoalescedTrainingOperationSet::isSetLeader(TrainingMatrixOperation* trainOp) {
    for(unsigned int i = 0; i < trainOps_.size(); ++i) {
        TrainingMatrixOperation* t = trainOps_[i];
        if(t != NULL) {
            return (t == trainOp); // Leader is first non-NULL MVM in the set
        }
    }
    assert(0 && "Unreachable: cannot have caolesced set with all NULL mvms!");
}

bool CoalescedTrainingOperationSet::isComplete() {
    for(auto trainOp : trainOps_) {
        if(trainOp == NULL) {
            return false;
        }
    }
    return true;
}

std::string Operation::printNodeName() {
    std::stringstream ss;
    ss << '"' << printOperationType() << "\n" << this << model_->printAssignment(this) << '"';
    return ss.str();
}

std::string Operation::printNodeStyle() {
    return "";
}

std::string MVMOperation::printNodeStyle() {
    return "[style=filled,fillcolor=\"#009933\"]";
}

std::string TrainingMatrixOperation::printNodeStyle() {
    return "[style=filled,fillcolor=\"#009933\"]";
}

std::string ALUVectorOperation::printNodeStyle() {
    return "[style=filled,fillcolor=\"#66FF66\"]";
}

std::string LoadOperation::printNodeStyle() {
    return "[style=filled,fillcolor=\"#FFB366\"]";
}

std::string StoreOperation::printNodeStyle() {
    return "[style=filled,fillcolor=\"#FFB366\"]";
}

std::string SendOperation::printNodeStyle() {
    return "[style=filled,fillcolor=\"#FFFF66\"]";
}

std::string ReceiveOperation::printNodeStyle() {
    return "[style=filled,fillcolor=\"#FFFF66\"]";
}

std::string MVMOperation::printOperationType() {
    std::stringstream ss;
    ss << "MVM: " << mat_->name();
    return ss.str();
}

std::string TrainingMatrixOperation::printOperationType() {
    std::stringstream ss;
    switch(opType_) {
        case MVM:           ss << "MVM";            break;
        case MVM_TRANSPOSE: ss << "MVM_TRANSPOSE";  break;
        case OUTER_PRODUCT: ss << "OUTER_PRODUCT";  break;
    }
    ss << ": " << mat_->name();
    return ss.str();
}

std::string ALUVectorOperation::printOperationType() {
    switch(opCode_) {
        case ADD: return "ADD";
        case SUB: return "SUB";
        case MUL: return "MUL";
        case DIV: return "DIV";
        case MULI: return "MULI";
        case AND: return "AND";
        case OR: return "OR";
        case NOT: return "NOT";
        case EQ: return "EQ";
        case NEQ: return "NEQ";
        case LT: return "LT";
        case LEQ: return "LEQ";
        case GT: return "GT";
        case GEQ: return "GEQ";
        case MIN: return "MIN";
        case MAX: return "MAX";
        case MSE: return "MSE";
        case SIG: return "SIG";
        case TANH: return "TANH";
        case EXP: return "EXP";
        case LOG: return "LOG";
        case RELU: return "RELU";
        case RELUD: return "RELUD";
        case LOG_SOFTMAX: return "LOG_SOFTMAX";
        case LOG_SOFTMAXD: return "LOG_SOFTMAXD";
        case RNDCMP: return "RNDCMP";
    }
}

std::string SetImmediateOperation::printOperationType() {
    std::stringstream ss;
    ss << "Set " << imm_;
    return ss.str();
}

std::string CopyOperation::printOperationType() {
    return "Copy";
}

std::string StoreOperation::printOperationType() {
    return "Store";
}

std::string LoadOperation::printOperationType() {
    return "Load";
}

std::string SendOperation::printOperationType() {
    return "Send";
}

std::string ReceiveOperation::printOperationType() {
    return "Receive";
}

std::string WriteInputOperation::printOperationType() {
    return "WriteInput";
}

std::string ReadOutputOperation::printOperationType() {
    return "ReadOutput";
}

std::string PseudoInputOperation::printOperationType() {
    return "PseudoInput";
}

std::string PseudoOutputOperation::printOperationType() {
    return "PseudoOutput";
}

void Operation::printNodeAndEdges(std::ostream& fout) {
    fout << printNodeName() << " " << printNodeStyle() << ";" << std::endl;
}

void ProducerOperation::printNodeAndEdges(std::ostream& fout) {
    Operation::printNodeAndEdges(fout);
    for(ConsumerOperation* user : users_) {
        fout << printNodeName() << " -> " << user->printNodeName() << ";" << std::endl;
    }
}

void TileMemoryWriteOperation::printNodeAndEdges(std::ostream& fout) {
    Operation::printNodeAndEdges(fout);
    for(TileMemoryReadOperation* user : users_) {
        fout << printNodeName() << " -> " << user->printNodeName() << ";" << std::endl;
    }
}

void SendOperation::printNodeAndEdges(std::ostream& fout) {
    Operation::printNodeAndEdges(fout);
    fout << printNodeName() << " -> " << dst_->printNodeName() << ";" << std::endl;
}

void InputOperation::printNodeAndEdges(std::ostream& fout) {
    fout << src_->printNodeName() << " -> " << printNodeName() << ";" << std::endl;
}

void OutputOperation::printNodeAndEdges(std::ostream& fout) {
    Operation::printNodeAndEdges(fout);
    fout << printNodeName() << " -> " << dst_->printNodeName() << ";" << std::endl;
}

void WriteInputOperation::printNodeAndEdges(std::ostream& fout) {
    TileMemoryWriteOperation::printNodeAndEdges(fout);
    InputOperation::printNodeAndEdges(fout);
}

void ReadOutputOperation::printNodeAndEdges(std::ostream& fout) {
    OutputOperation::printNodeAndEdges(fout);
}


void PseudoInputOperation::printNodeAndEdges(std::ostream& fout) {
    ProducerOperation::printNodeAndEdges(fout);
    InputOperation::printNodeAndEdges(fout);
}

void PseudoOutputOperation::printNodeAndEdges(std::ostream& fout) {
    OutputOperation::printNodeAndEdges(fout);
}

