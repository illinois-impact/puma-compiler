/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _COMMON_H_
#define _COMMON_H_

#include "puma.h"

/* Constants */
#define MVMU_DIM                        128
#define N_CONSTANT_MVMUS_PER_CORE       6
#define N_TRAINING_MVMUS_PER_CORE       2
#define N_CORES_PER_TILE                8
#define MAX_LOAD_STORE_WIDTH            16
#define MAX_SEND_RECV_WIDTH             16
#define N_TRAINING_OPERATIONS           3
#define N_INPUT_REGISTERS               (MVMU_DIM*((N_CONSTANT_MVMUS_PER_CORE >= N_TRAINING_OPERATIONS*N_TRAINING_MVMUS_PER_CORE)?N_CONSTANT_MVMUS_PER_CORE:(N_TRAINING_OPERATIONS*N_TRAINING_MVMUS_PER_CORE)))
#define N_OUTPUT_REGISTERS              N_INPUT_REGISTERS
#define INPUT_REGISTERS_START_ADDRESS   0
#define OUTPUT_REGISTERS_START_ADDRESS  (INPUT_REGISTERS_START_ADDRESS + N_INPUT_REGISTERS)
#define REGISTER_FILE_START_ADDRESS     (OUTPUT_REGISTERS_START_ADDRESS + N_OUTPUT_REGISTERS)
#define REGISTER_FILE_SIZE              (N_INPUT_REGISTERS + N_OUTPUT_REGISTERS)
#define REGISTERS_PER_CORE              (N_INPUT_REGISTERS + N_OUTPUT_REGISTERS + REGISTER_FILE_SIZE)

/* tensors.h */
class AbstractTensor;
class AbstractVector;
class AbstractMatrix;
class AbstractImagePixelStream;
class InputVectorTile;
class InputVectorImpl;
class InputImagePixelStreamImpl;
class VectorImpl;
class ImagePixelStreamImpl;
class OutputVectorTile;
class OutputVectorImpl;
class OutputImagePixelStreamImpl;
class ConstantMatrixTile;
class ConstantMatrixImpl;
class ConvolutionalConstantMatrixImpl;
class TrainingMatrixTile;
class TrainingMatrixImpl;

/* operations.h */
class Operation;
class ProducerOperation;
class ConsumerOperation;
class TileMemoryWriteOperation;
class TileMemoryReadOperation;
class InputOperation;
class OutputOperation;
class CoreOperation;
class TileOperation;
class MVMOperation;
class CoalescedMVMSet;
class TrainingMatrixOperation;
class CoalescedTrainingOperationSet;
class ALUVectorOperation;
class SetImmediateOperation;
class CopyOperation;
class LoadOperation;
class StoreOperation;
class SendOperation;
class ReceiveOperation;
class WriteInputOperation;
class ReadOutputOperation;
class PseudoInputOperation;
class PseudoOutputOperation;

/* allocator.h */
class CoreAllocator;
class SpillTracker;

/* model.h */
class ModelImpl;

/* partitioner.h */
class Partitioner;

/* placer.h */
class Placer;

/* memalloc.h */
class MemoryAllocator;

/* coalescer.h */
class Coalescer;

/* linearizer.h */
class Linearizer;

/* regalloc.h */
class RegisterAllocator;

/* codegen.h */
class CodeGenerator;

/* instance.h */
class ModelInstanceImpl;

#endif

