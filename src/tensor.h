#pragma once


#include "util.h"
#include <iostream>
#include <string>
#include <cassert>


// Forward declare types.
struct Tensor;


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
 * Create a tensor scalar value.
 */
Tensor tensor_create_scalar(float value);


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
 * Initialze tensor 2D tensor as an identity matrix.
 */
void tensor_init_identity(Tensor& tensor);


/**
 * Matrix multiplication of two tensors. 
 * The result is stored in the left hand side tensor.
 */
Tensor tensor_matmul(Tensor& lhs, Tensor& rhs);


/**
 * Reshape the tensor from one shape to another.
 */
void tensor_reshape(Tensor& tensor, u32 shape[], u8 ndim);


/**
 * Checks that the shapes of two tensors to check if they are the same.
 */
inline void tensor_check_same_shape(Tensor& lhs, Tensor& rhs) {
#ifdef DEBUG
    // First check that there are equal number of elements, if the are different
    // then there is no way these tensors have the same shape.
    if (lhs.length != rhs.length) {
        std::cerr << "error: expected lhs and rhs to have same number of elements" << std::endl;
        assert(false);
    }

    // Check the dimensions, even if the mismatch there is still a change 
    // e.g. a 2d-tensor of shape 2x1 should be the same as a 1d-tensor of shape 2
    // We want to avoid unnecessary reshaping (i.e. memory packings and reallocations)
    // we can ignore dimensions with only one element.
    if (lhs.ndim != rhs.ndim) {
        u8 ldim = 0;
        for (int i = 0; i < lhs.ndim; i++) {
            if (lhs.shape[i] != 1) ldim++;
        }

        u8 rdim = 0;
        for (int i = 0; i < rhs.ndim; i++) {
            if (rhs.shape[i] != 1) rdim++;
        }

        if (ldim != rdim) {
            std::cerr << "error: expected lhs and rhs to have the same dimension" << std::endl;
            assert(false);
        }
    }

    // This check is rather silly but we want to check that the shapes of the tensors
    // match and ignore whenever there is fewer than 2 elements in a dimension.
    int i = 0;
    int j = 0;
    while (i <= lhs.ndim && j <= rhs.ndim) {
        if (lhs.shape[i] < 2) {
            i++;
            continue;
        }
        if (rhs.shape[j] < 2) {
            j++;
            continue;
        }
        if (lhs.shape[i] != rhs.shape[j]) {
            std::cerr << "error: tensor shape mismatch, dimension (lhs.shape[" << i << "] = " << lhs.shape[i] << ") ";
            std::cerr << "(rhs.shape[" << j << "] = " << rhs.shape[j] << ")" << std::endl;
            assert(false);
        }
        i++;
        j++;    
    }
#endif
}


/**
 * Checks that the shapes of two tensors to check if they 
 * are applicable for matrix multiplication.
 */
inline void tensor_check_matmul_shape(Tensor& lhs, Tensor& rhs) {
#ifdef DEBUG
    assert(lhs.ndim == 2);
    assert(rhs.ndim == 2);
    assert(lhs.shape[0] == rhs.shape[1]);
#endif
}


/**
 * Element-wise tensor addition. 
 * The result is stored in the left hand side tensor.
 */
inline void tensor_add(Tensor& lhs, Tensor& rhs) {
    tensor_check_same_shape(lhs, rhs);
    for (int i = 0; i < lhs.length; i++) {
        lhs.data[i] += rhs.data[i];
    }
}


/**
 * Element-wise tensor addition.
 * The result is stored in a new tensor, this is pure.
 */
inline Tensor tensor_copy_add(Tensor& lhs, Tensor& rhs) {
    Tensor lhs_copy = tensor_copy(lhs);
    tensor_add(lhs_copy, rhs);
    return lhs_copy;
}


/**
 * Element-wise tensor multiplication. 
 * The result is stored in the left hand side tensor.
 */
inline void tensor_mul(Tensor& lhs, Tensor& rhs) {
    tensor_check_same_shape(lhs, rhs);
    for (int i = 0; i < lhs.length; i++) {
        lhs.data[i] *= rhs.data[i];
    }
}


/**
 * Element-wise tensor multiplication.
 * The result is stored in a new tensor, this is pure.
 */
inline Tensor tensor_copy_mul(Tensor& lhs, Tensor& rhs) {
    Tensor lhs_copy = tensor_copy(lhs);
    tensor_mul(lhs_copy, rhs);
    return lhs_copy;
}


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
