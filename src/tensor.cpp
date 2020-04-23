#include "util.h"
#include "tensor.h"
#include <cmath>
#include <iomanip>
#include <iostream>


Tensor tensor_create_1d(u32 len) {
    Tensor tensor;
    tensor.ndim = 1;
    tensor.length = len;
    tensor.shape = (u32*) malloc(sizeof(u32)*tensor.ndim);
    tensor.shape[0] = len;
    tensor.data = (float*) malloc(sizeof(float)*tensor.length);
    return tensor;
}


Tensor tensor_create_2d(u32 xlen, u32 ylen) {
    Tensor tensor;
    tensor.ndim = 2;
    tensor.length = xlen*ylen;
    tensor.shape = (u32*) malloc(sizeof(u32)*tensor.ndim);
    tensor.shape[0] = xlen;
    tensor.shape[1] = ylen;
    tensor.data = (float*) malloc(sizeof(float)*tensor.length);
    return tensor;
}

Tensor tensor_init_identity(u32 xlen, u32 ylen) {
    Tensor tensor = tensor_create_2d(xlen, ylen);
    tensor_init_zeros(tensor);
    for (int i = 0; i < fmin(xlen, ylen); i++) {
        tensor.data[(i % xlen) + (i / xlen)] = 1;
    }
    return tensor;
}


Tensor tensor_copy(Tensor& tensor) {
    Tensor copy;
    copy.ndim = tensor.ndim;
    copy.length = tensor.length;
    copy.shape = (u32*) malloc(sizeof(u32)*copy.ndim);
    copy.data = (float*) malloc(sizeof(float)*copy.length);
    memcpy(copy.shape, tensor.shape, sizeof(u32)*copy.ndim);
    memcpy(copy.data, tensor.data, sizeof(float)*copy.length);
    return copy;
}


void tensor_init_zeros(Tensor& tensor) {
    for (int i = 0; i < tensor.length; i++) {
        tensor.data[i] = 0;
    }
}



void tensor_init_ones(Tensor& tensor) {
    for (int i = 0; i < tensor.length; i++) {
        tensor.data[i] = 1;
    }
}

void tensor_init_random(Tensor& tensor) {
    for (int i = 0; i < tensor.length; i++) {
        tensor.data[i] = (float) rand() / RAND_MAX;
    }
}


inline void tensor_add(Tensor& lhs, Tensor& rhs) {
    tensor_check_same_shape(lhs, rhs);
    for (int i = 0; i < lhs.length; i++) {
        lhs.data[i] += rhs.data[i];
    }
}


inline Tensor tensor_copy_add(Tensor& lhs, Tensor& rhs) {
    Tensor lhs_copy = tensor_copy(lhs);
    tensor_add(lhs_copy, rhs);
    return lhs_copy;
}


inline void tensor_mul(Tensor& lhs, Tensor& rhs) {
    tensor_check_same_shape(lhs, rhs);
    for (int i = 0; i < lhs.length; i++) {
        lhs.data[i] *= rhs.data[i];
    }
}


inline Tensor tensor_copy_mul(Tensor& lhs, Tensor& rhs) {
    Tensor lhs_copy = tensor_copy(lhs);
    tensor_mul(lhs_copy, rhs);
    return lhs_copy;
}


inline void tensor_matmul(Tensor& lhs, Tensor& rhs) {
    tensor_check_matmul_shape(lhs, rhs);
    for (int i = 0; i < lhs.length; i++) {
        lhs.data[i] *= rhs.data[i];
    }
}


inline Tensor tensor_copy_matmul(Tensor& lhs, Tensor& rhs) {
    Tensor lhs_copy = tensor_copy(lhs);
    tensor_mul(lhs_copy, rhs);
    return lhs_copy;
}


inline void tensor_check_same_shape(Tensor& lhs, Tensor& rhs) {
#ifdef DEBUG
    assert(lhs.ndim == rhs.ndim);
    assert(lhs.length == rhs.length);
    for (int i = 0; i < lhs.ndim; i++) {
        assert(lhs.shape[i] == rhs.shape[i]);
    }
#endif
}


inline void tensor_check_matmul_shape(Tensor& lhs, Tensor& rhs) {
#ifdef DEBUG
    assert(lhs.ndim == 2);
    assert(rhs.ndim == 2);
    assert(lhs.shape[0] == rhs.shape[1]);
#endif
}


std::ostream& operator<<(std::ostream& stream, Tensor& tensor) {
    stream << std::fixed << std::setprecision(2);
    stream << "┌";
    for (int i = 0; i < tensor.shape[0]; i++) stream << "      ";
    stream << "┐" << std::endl;

    for (int i = 0; i < tensor.length; i++) {
        if (i % tensor.shape[0] == 0) {
            stream << "| ";
        }
        stream << tensor.data[i] << " ";
        if ((i % tensor.shape[0]) == (tensor.shape[0] - 1)) {
            stream << "|" << std::endl;
        } else {
            stream << " ";
        }
    }

    stream << "└";
    for (int i = 0; i < tensor.shape[0]; i++) stream << "      ";
    stream << "┘" << std::endl;
    return stream;
}
