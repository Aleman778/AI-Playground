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


void tensor_init_identity(Tensor& tensor) {
    assert(tensor.ndim == 2);
    tensor_init_zeros(tensor);

    u32 xlen = tensor.shape[0];
    u32 ylen = tensor.shape[1];
    for (int i = 0; i < fmin(xlen, ylen); i++) {
        tensor.data[i*xlen + i] = 1;
    }
}


Tensor tensor_matmul(Tensor& lhs, Tensor& rhs) {
    tensor_check_matmul_shape(lhs, rhs);
    u32 xlen_lhs = lhs.shape[0];
    u32 ylen_lhs = lhs.shape[1];
    u32 xlen_rhs = rhs.shape[0];
    Tensor out = tensor_create_2d(xlen_rhs, ylen_lhs);
    tensor_init_zeros(out);
    for (int j = 0; j < ylen_lhs; j++) {
        for (int i = 0; i < xlen_rhs; i++) {
            for (int k = 0; k < xlen_lhs; k++) {
                out.data[i + j*xlen_rhs] += lhs.data[k + j*xlen_lhs]*rhs.data[i + k*xlen_rhs];
            }
        }
    }
    return out;
}


void tensor_reshape(Tensor& tensor, u32 shape[], u8 ndim) {
    u32 newLength = 1;

    free(tensor.shape);
    tensor.shape = (u32*) malloc(sizeof(u32)*ndim);
    for (u8 i = 0; i < ndim; i++) {
        newLength *= shape[i];
        tensor.shape[i] = shape[i];
    }

    if (newLength != tensor.length) {
        float* newData = (float*) realloc((void*) tensor.data, sizeof(float)*newLength);
        if (!newData) {
            free(tensor.data);
            tensor.data = (float*) malloc(sizeof(float)*newLength);
            memcpy(newData, tensor.data, sizeof(float)*newLength);
        }
        tensor.data = newData;
    }
    std::cout << newLength << std::endl;

    tensor.ndim = ndim;
    tensor.length = newLength;
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
