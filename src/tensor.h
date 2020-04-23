#pragma once


#include "util.h"
#include <iostream>
#include <string>
#include <cassert>


// Forward declare types.
struct Tensor;
struct CPU_Tensor;


/**
 * Abstract tensor provides an interface to operate on tensors that
 * could possibly be stored on the CPU or on another device.
 * Using to(device), cpu(), or cuda() function will transfer the data from
 * one device to another.
 */
struct Tensor {
    // Number of dimensions.
    u8 ndim;
    // Number of elements in tensor.
    u32 length;
    // Number of elements per dimension.
    u32* shape;
    // Pointer to the actual storage.
    float* data;
};


/**
 * Creates a new 1-dimensional tensor of specific length.
 */
Tensor tensor_create_1d(u32 len);


/**
 * Creates a new 2-dimensional tensor of specific length.
 */
Tensor tensor_create_2d(u32 xlen, u32 ylen);


/**
 * Makes a copy of the provided tensor pointer.
 */
Tensor tensor_copy(Tensor& tensor);


/**
 * Initialize tensor with zeros.
 */
void tensor_init_zeros(Tensor& tensor);


/**
 * Initialize tensor with ones.
 */
void tensor_init_ones(Tensor& tensor);


/**
 * Initialize tensor with random numbers.
 */
void tensor_init_random(Tensor& tensor);


/**
 * Element-wise tensor addition. 
 * The result is stored in the left hand side tensor.
 */
void tensor_add(Tensor& lhs, Tensor& rhs);


/**
 * Element-wise tensor addition.
 * The result is stored in a new tensor, this is pure.
 */
Tensor tensor_copy_add(Tensor& lhs, Tensor& rhs);


/**
 * Element-wise tensor multiplication. 
 * The result is stored in the left hand side tensor.
 */
void tensor_mul(Tensor& lhs, Tensor& rhs);


/**
 * Element-wise tensor multiplication.
 * The result is stored in a new tensor, this is pure.
 */
Tensor tensor_copy_mul(Tensor& lhs, Tensor& rhs);


/**
 * Matrix multiplication of two tensors. 
 * The result is stored in the left hand side tensor.
 */
void tensor_matmul(Tensor& lhs, Tensor& rhs);


/**
 * Matrix multiplication of two tensors.
 * The result is stored in a new tensor, this is pure.
 */
Tensor tensor_copy_matmul(Tensor& lhs, Tensor& rhs);


/**
 * Checks that the shapes of two tensors to check if they are the same.
 */
void tensor_check_same_shape(Tensor& lhs, Tensor& rhs);


/**
 * Checks that the shapes of two tensors to check if they 
 * are applicable for matrix multiplication.
 */
void tensor_check_matmul_shape(Tensor& lhs, Tensor& rhs);


/**
 * Operator overloading for element-wise tensor addition.
 */
inline Tensor operator+(Tensor& lhs, Tensor& rhs) {
    return tensor_copy_add(lhs, rhs);
}


/**
 * Operator overloading for element-wise tensor addition.
 */
inline Tensor operator*(Tensor& lhs, Tensor& rhs) {
    return tensor_copy_mul(lhs, rhs);
}

/**
 * Render the tensor to output stream.
 */
std::ostream& operator<<(std::ostream& stream, Tensor& tensor);
