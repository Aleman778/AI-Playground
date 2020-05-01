#pragma once


#include "tensor.h"
#include <cmath>


/***************************************************************************
 * Activation functions
 ***************************************************************************/


inline void f_relu(Tensor& tensor) {
    for (int i = 0; i < tensor.length; i++) {
        tensor.data[i] = fmax(0.0f, tensor.data[i]);
    }
}
