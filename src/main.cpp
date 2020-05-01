#include "tensor.h"
#include "network.h"
#include <iostream>


void init_incr(Tensor& t) {
    for (int i = 0; i < t.length; i++) t.data[i] = i + 1;
}


int main(int argc, char** argv) {

    Network model = sequential({
        dense_layer(3, 2, init_incr, tensor_init_zeros),
        relu(),
    });
        

    Tensor input = tensor_create_2d(1, 3);
    tensor_init_random(input);


    Tensor output = model.forward(&input);

    // std::cout << "Weights:\n" << layer.weights << std::endl;
    // std::cout << "Biases:\n" << layer.bias << std::endl;
    std::cout << "Input:\n" << input << std::endl;
    std::cout << "Output:\n" << output << std::endl;

    return 0;
}
