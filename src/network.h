#pragma once


#include "tensor.h"
#include "functional.h"
#include <initializer_list>
#include <vector>


// Forward declare types.
struct Network;
struct Layer;
struct Dense_Layer;
struct ReLU_Layer;


/**
 * Layer represents a single layer of neurons that is together with
 * many layers forms a neural network. The layers does not have to be
 * neurons but can also be for instance a convolution- or activation layer.
 */
struct Layer {
    /// Requires the gradient to be calculated in backwards step.
    bool require_grad = true;

    virtual Tensor forward(Tensor* input) = 0;
};


/**
 * Dense layer connects all the input neurons 
 * in this layer to every output neuron. 
 * The output neuron can be feed-forward to another layer or module.
 */
struct Dense_Layer : Layer {
    Tensor weights;
    Tensor bias;

    virtual Tensor forward(Tensor* input) override {
        Tensor output = tensor_matmul(weights, *input);
        tensor_add(output, bias);
        return output;
    }
};


/**
 * Rectified Linear Unit is an activiation function is defined
 * as `fmax(0.0f, x)` for each element `x` in the input tensor.
 */
struct ReLU_Layer : Layer {
    bool inplace = false;
    
    virtual Tensor forward(Tensor* input) override {
        Tensor t;
        if (inplace) {
            t = *input;
        } else {
            t = tensor_copy(*input);
        }
        f_relu(t);
        return t;
    }
};


/**
 * Neural network defiend by an array of layers.
 */
struct Network {
    std::vector<Layer> layers;


    /**
     * Forwards the input through the list of layers and
     * returns the resulting tensor.
     */
    Tensor forward(Tensor* input) {
        Tensor output = *input;
        for (int i = 0; i < layers.size(); i++) {
            output = layers[i].forward(&output);
        }
        return output;
    }
};


/**
 * Creates a neural network with layers in sequence.
 */
inline Network sequential(std::initializer_list<Layer> layers) {
    Network network;
    network.layers = layers;
    return network;
}

    

/**
 * Create a new dense layer which connects every neuron
 * to each neuron in the next layer.
 */
Dense_Layer dense_layer(
    int num_inputs, 
    int num_outputs, 
    void init_weights(Tensor&),
    void init_biases(Tensor&)
);


inline ReLU_Layer relu(bool inplace = false) {
    ReLU_Layer layer;
    layer.inplace = inplace;
    return layer;
}
    
