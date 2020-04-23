#include "tensor.h"
#include <iostream>


int main(int argc, char** argv) {
    Tensor weights = tensor_create_2d(3, 3);
    Tensor input = tensor_create_2d(3, 3);

    tensor_init_random(weights);
    tensor_init_random(input);

    Tensor output = weights + input;

    std::cout << output << std::endl;
    return 0;
}
