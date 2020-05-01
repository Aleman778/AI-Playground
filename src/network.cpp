#include "network.h"
#include "tensor.h"


Dense_Layer dense_layer(
    int num_inputs, 
    int num_outputs, 
    void init_weights(Tensor&) = tensor_init_random,
    void init_biases(Tensor&) = tensor_init_random
) {
    Dense_Layer layer;
    layer.weights = tensor_create_2d(num_inputs, num_outputs);
    layer.bias = tensor_create_1d(num_outputs);
    init_weights(layer.weights);
    init_biases(layer.bias);
    return layer;
}
