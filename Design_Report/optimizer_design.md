# Optimizer Design



## 1. Overview

### Optimizer Module

The `Optimizer` module provides essential tools for updating model parameters during training by implementing commonly used optimizers like Stochastic Gradient Descent (SGD) and Adam.



### Optimizer Structure

```ocaml
type t = {
  parameters: tensor list;  (* List of tensors representing the parameters to optimize. *)
  step: unit -> unit;       (* Function to update parameters based on their gradients. *)
  zero_grad: unit -> unit;  (* Function to reset all gradients to zero. *)
}

```



Each optimizer instance consists of:

- **`parameters`**: A list of tensors (parameters) that the optimizer will update based on computed gradients.
- **`step`**: A function to execute one optimization step, updating parameters according to their gradients.
- **`zero_grad`**: A function to reset gradients of all parameters to zero, preparing them for the next backpropagation pass.

### Supported Optimizers

1. **SGD (Stochastic Gradient Descent)**: Updates parameters by moving them in the direction of the gradient, scaled by a learning rate.
2. **Adam**: A more sophisticated optimizer that adapts the learning rate for each parameter based on moment estimates, enhancing convergence in many scenarios.



## 2. Mock Use

Below is an example of how to use the `Optimizer` module to train a model's parameters.

### Example Usage

```ocaml
(* Initialize model parameters and create an optimizer *)

(* Step 1: Create sample tensors representing model parameters *)
let param1 = Tensor.create ~data:[|0.5; -0.3|] ~shape:[|2|] ~requires_grad:true in
let param2 = Tensor.create ~data:[|1.0; 0.8|] ~shape:[|2|] ~requires_grad:true in
let parameters = [param1; param2] in

(* Step 2: Initialize an SGD optimizer *)
let optimizer = create_SGD ~params:parameters ~lr:0.01 in

(* Step 3: Perform a forward and backward pass, then update parameters *)
let loss = ... (* Compute loss based on forward pass *) in
Tensor.backward loss;  (* Compute gradients via backpropagation *)

(* Step 4: Update parameters and reset gradients *)
step optimizer;
zero_grad optimizer;

```

In this example:

1. **Parameter Initialization**: Two tensors representing parameters (`param1` and `param2`) are created with gradient tracking enabled.
2. **Optimizer Creation**: An SGD optimizer is initialized with these parameters and a learning rate of `0.01`.
3. **Forward and Backward Pass**: The model’s forward pass computes a loss, followed by `Tensor.backward` to calculate gradients for each parameter.
4. **Parameter Update and Gradient Reset**: The `step` function updates each parameter based on its gradient, and `zero_grad` clears the gradients, preparing for the next iteration.

This setup simulates a typical training loop where parameters are updated iteratively until the model converges on a solution.



## 3. Library Dependencies

The `Optimizer` module relies on the following dependencies:

- **Tensor**: Provides gradient tracking and enables parameter updates by supporting automatic differentiation.



## 4. Implementation Plan

The implementation plan for the `Optimizer` module is as follows:

**This week**: **Jixiao Zhang** and **Suizhi Ma** will implement SGD and Adam optimizers, including parameter tracking, gradient updates, and learning rate configurations.

**Next week**: **Rui Wang** and **Chuan Chen** will test optimizer behaviors with both simple models and complex neural networks to ensure correct parameter updates.