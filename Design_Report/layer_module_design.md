# Layer Module Design



## 1. Overview

### Neural Network Layer Module

This module defines a structure for implementing neural network layers and modules. The primary components include parameterized layers (e.g., linear, convolutional) and activation functions (e.g., ReLU, Sigmoid), as well as essential loss functions like Mean Squared Error (MSE) and Cross Entropy.

### Layer Structure

```ocaml
type t = {
  parameters: tensor list;          
  (* List of tensors representing the layer's parameters. *)
  
  forward_fn: tensor list -> tensor; 
  (* Function to compute the forward pass of the layer. *)
}

```



Each layer includes:

- **`parameters`**: A list of tensors that store the layer's learnable parameters (e.g., weights, biases).
- **`forward_fn`**: A function that defines how to perform the forward pass given a list of input tensors. This function is specific to the layer type and calculates the output based on the input and layer parameters.



### Key Layer Functions

The module provides essential functions for interacting with layers:

- **`forward`**: Executes a forward pass for a given layer with specified inputs.
- **`get_parameters`**: Retrieves the learnable parameters from a layer for use in optimization.
- **Layer Creators**: Factory functions for creating commonly used layers (e.g., Linear, Conv2d) and activation functions (e.g., ReLU, Sigmoid), as well as loss layers like MSE and Cross Entropy.



## 2. Mock Use

The following example demonstrates how to create a simple neural network with a few layers, perform a forward pass, and retrieve the output.

### Example Usage

```ocaml
(* Initialize a simple neural network with a linear layer, activation, and another linear layer *)

(* Step 1: Create layers *)
let layer1 = create_Linear ~in_features:4 ~out_features:3 ~bias:true in
let relu = create_ReLU () in
let layer2 = create_Linear ~in_features:3 ~out_features:2 ~bias:true in

(* Step 2: Create a Sequential container *)
let model = create_Sequential [layer1; relu; layer2] in

(* Step 3: Perform a forward pass *)
let input_data = Tensor.of_array [|1.0; 2.0; 3.0; 4.0|] ~shape:[|1; 4|] in
let output = forward model [input_data] in

(* Output the result *)
Printf.printf "Network output: %s\n" (Tensor.to_string output);
```



In this example:

1. **Layer Creation**: The network is initialized with a sequence of layers—a `Linear` layer, a `ReLU` activation, and another `Linear` layer.
2. **Sequential Model**: The layers are combined in a sequential container, `model`, allowing each layer to pass its output as input to the next layer.
3. **Forward Pass**: An input tensor `input_data` is passed through the `model` via `forward`, producing an output tensor.
4. **Output**: The output of the network is printed, demonstrating the model's response to the input data.



## 3. Library Dependencies

The `Layer` module relies on the following libraries:

- **Tensor**: For managing tensor operations and gradients.
- **Ndarray**: Underlying multidimensional array structure for efficient data storage and computation.



## 4. Implementation Plan

Our plan to implement the `Layer` module consists of the following steps:

**This Week**: **Suizhi Ma** and **Jixiao Zhang** will implement core layers (e.g., Linear, Conv2d, Flatten) and activation functions (e.g., ReLU, Sigmoid). Each layer will have a corresponding forward function to compute its output.

**Next week**: **Suizhi Ma** and **Jixiao Zhang** Develop the Sequential container and loss layers (e.g., MSE, Cross Entropy). This includes defining loss calculations for training and backpropagation.

**Next next week**: **Everyone** will do it together, we will integrate forward and backward passes, ensuring that each layer records its predecessors and gradient computations. We will also add utility functions to retrieve layer parameters for model training.