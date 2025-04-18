# OCaMLearn (OCaml machine learning framework)

## Overview

Our project aims to develop a machine learning library in OCaml, inspired by PyTorch, to provide the essential components for building and training machine learning models. This includes support for multi-dimensional arrays, tensors with automatic differentiation, and tools for defining and optimizing neural networks. The library focuses on modularity, usability, and flexibility, enabling users to create, customize, and train machine learning models for a variety of tasks, such as image classification.

## How to build and run it?
Before building our library, you should first install package ``csv`` using
```
opam install csv
```
Run the following command at the root directory of this repository to build and test our library.
```
dune build
dune test
```
Then, you can also try our demo in ``src/bin/main.ml`` using
```
dune exec src/bin/main.exe
```

## How to use it?
Please check our ``.mli`` files in ``src/lib/`` for your reference. All of the functions are well documented. You can also refer to our demo code in ``src/bin/main.ml`` which gives several examples on how to load data, build and train models. To understand the usage of each module, check the following module design and mock uses.

## Test & Coverage reports
You can check our coverage reports in ``_coverage`` or generate new ones using
```
dune clean
dune test --instrument-with bisect_ppx
bisect-ppx-report html
```

## Module Design

### Ndarray Design

#### 1. Overview

Our project is designed with `ndarray` and `tensor` as the fundamental computational modules.

##### `Ndarray` Module

The `ndarray` module is defined as follows:

```ocaml
type t = {
  data: float array;
  shape: int array;
}
```

In this structure, we use a one-dimensional array `data` to store all values, and an integer array `shape` to represent the dimensions of the `ndarray`.

To support operations between `ndarray`s with different shapes, we implemented broadcasting, which enables element-wise addition, subtraction, multiplication, division, as well as matrix multiplication.

We also provided multiple initialization methods, including custom values, as well as Xavier and Kaiming initialization for more advanced applications.

To facilitate usage, we defined `set` and `at` methods for fast access and modification of elements within the `ndarray`.

For computations, we implemented functions like `max`, `min`, and `mean`, along with methods to compute these statistics along specified axes.

Finally, we added functions like `slice` and `squeeze` to manipulate and reshape the `ndarray` efficiently.

With these definitions, we aim to create an API similar to Python's `numpy`, providing a familiar and convenient interface for numerical operations in OCaml.

#### 2. Mock Use

We conducted comprehensive testing of the module in `tests.ml`. Below is an example test for the `max` function:

```ocaml
(* Test for max function *)
let test_max () =
  Printf.printf "Testing max function...\n";

  (* Initialize data *)
  let data = [|1.0; 3.0; 5.0; 2.0; 4.0; 6.0|] in
  let shape = [|2; 3|] in

  (* Create ndarray *)
  let t = create data shape in

  (* Execute max operation *)
  let result = max t in
  let expected = 6.0 in

  (* Assert result *)
  assert (result = expected);
  Printf.printf "Passed: Max Array Data\n";
;;
```

In this test, we initialize `data` and `shape`, then use `create` to generate an `ndarray`. The `max` function is called to find the maximum value, and an assertion is used to verify the result.



### Tensor Design

#### 1. Overview

##### `Tensor` Module

The `Tensor` module builds on top of `ndarray`, enabling gradient tracking and backpropagation capabilities essential for machine learning.

```ocaml
type ndarray = Ndarray.t

type ndarray = Ndarray.t

type t = {
  mutable data: ndarray;
  mutable grad: ndarray;
  requires_grad: bool;
  mutable backward_fn: (unit -> unit) option;
  prev: t list;
}
```

In the `Tensor` structure:

- **`data`**: Stores the actual tensor data as an `ndarray`.
- **`grad`**: Stores the gradient of `data`, which can be computed during backpropagation if `requires_grad` is `true`.
- **`requires_grad`**: Indicates whether gradient computation is necessary. If `true`, gradients are stored in `grad`.
- **`backward_fn`**: Defines the backpropagation function for the tensor, specifying how gradients should be computed.
- **`prev`**: Stores a list of previous tensors, allowing construction of the computation graph for backpropagation.

The `Tensor` module allows for gradient-based optimization by constructing computation graphs and enabling automatic differentiation, similar to the functionality in Python’s PyTorch library.

#### 2. Mock Use

We have written comprehensive tests for both `ndarray` and `Tensor` functionalities in `tests.ml`. Below is an example of a test for the `max` function in the `Tensor` module:

```ocaml
(* Test for element-wise tensor multiplication *)
let test_tensor_multiplication () =
  Printf.printf "Testing tensor multiplication...\n";

  (* Initialize data for two tensors *)
  let data1 = [|1.0; 2.0; 3.0; 4.0|] in
  let shape1 = [|2; 2|] in
  let t1 = create ~data:data1 ~shape:shape1 ~requires_grad:false in

  let data2 = [|5.0; 6.0; 7.0; 8.0|] in
  let shape2 = [|2; 2|] in
  let t2 = create ~data:data2 ~shape:shape2 ~requires_grad:false in

  (* Perform element-wise multiplication *)
  let result = mul t1 t2 in

  (* Expected result *)
  let expected_data = [|19.0; 22.0; 43.0; 50.0|] in
  let expected_shape = [|2; 2|] in

  (* Assertions *)
  assert (result.data = expected_data);
  assert (result.shape = expected_shape);
  Printf.printf "Passed: Tensor Multiplication\n";
;;

```

In this example:

1. We initialize two tensors `t1` and `t2` with `data` and `shape` values.
2. We use the `mul` function to perform element-wise multiplication, resulting in a new tensor `result`.
3. Assertions check that the resulting tensor `result` matches the expected `data` and `shape`.
4. Additionally, and most importantly, through the `mul` function, we create a new tensor `result` by multiplying `t1` and `t2`. The `result` tensor keeps track of its two predecessors, `t1` and `t2`, and stores a function for backpropagation. This setup enables `result` to propagate gradients back to `t1` and `t2` during backpropagation, allowing gradients to flow through the computation graph.



### Dataset & Dataloader & Transform Design

#### 1. Overview

##### `Dataset` Module

The `Dataset` module is designed to process and manipulate data samples and their corresponding labels. It provides basic functions such as data retrieval, shuffling, and splitting, which are common requirements in data preprocessing workflows.

##### Dataset Structure

```ocaml
type t = {
  data : ndarray; 
  label : ndarray; 
}

```

In the `Dataset` structure:

- **`data`**: Stores the data samples in the form of an `ndarray`.
- **`label`**: Stores the corresponding labels for each data sample in the dataset.

This structure ensures that each data sample is directly associated with a label, allowing for efficient data access and manipulation.

##### Key Functionalities

The `Dataset` module provides the following core functionalities:

- **`get_item`**: Retrieves a specific data sample and its label by index.
- **`shuffle`**: Returns a new dataset with randomly shuffled samples.
- **`split`**: Divides the dataset into two parts based on a specified ratio, often used to create training and validation sets.




##### `DataLoader` Module

The `DataLoader` module is designed to manage and organize data batching, shuffling, and optional data transformations for efficient training workflows in deep learning. It provides functions that enable seamless retrieval of data batches for iterative model training, with options for both sequential and randomized access.

##### DataLoader  Structure

```ocaml
type t = {
  dataset : tensor_dataset; 
  batch_size : int; 
  total_batches : int;
}

```

In the `DataLoader` structure:

- **`dataset`**: Stores the dataset of tensors, consisting of both data samples and labels, for loading in batches.
- **`batch_size`**: Specifies the number of data samples per batch, controlling the unit of data fed to the model at each step.
- **`total_batches`**: Indicates the total number of batches available in the dataset, based on the batch size and the dataset size.
This structure ensures that each data sample is directly associated with a label, allowing for efficient data access and manipulation.

##### Key Functionalities

The `DataLoader` module provides the following core functionalities:

- **`create`**: Initializes a new data loader instance with configurable batch size, shuffling options, and optional transformations.
- **`get_batch`**: Retrieves a specific batch of data samples and labels by batch index, enabling controlled access to data during model training.
- **`get_total_batches`**: Returns the total number of batches available, facilitating iteration planning and tracking within training loops.

##### `Transform` Module

The `Transform` module provides a set of operations designed to preprocess and augment tensors. These transformations are crucial in preparing data to match the model's requirements and for introducing variations that improve model generalization. They can be applied dynamically during data loading or preprocessing stages.

##### Transform Structure

```ocaml
type t = tensor -> tensor
(* A transformation is a function that takes a tensor as input and returns a transformed tensor. *)
```

The `Transform` structure allows for modular and composable operations. Each transformation function adheres to the same type signature, making them easy to integrate into data pipelines.

##### Key Functionalities

The `Transform` module provides the following core transformations:

- **`normalize`**: Standardizes the tensor to have a mean of 0 and standard deviation of 1, preparing it for models sensitive to scale.
<!-- - **`resize`**: Adjusts the dimensions of the tensor to the desired width and height.
- **`rotate`**: Rotates the tensor by a specified angle, introducing rotational variations.
- **`translate`**: Shifts the tensor along the x and y axes by given offsets. -->
- **`image_scale`**: Resizes the tensor dynamically by applying scaling factors along the x and y dimensions.
<!-- - **`flip`**: Flips the tensor horizontally, vertically, or both, enhancing data variability. -->

---

#### 2. Mock Use

The following will show how to use the `Dataset` and `Dataloader` modules.

##### Example Usage

```ocaml
(* Step 1: Initialize the Dataset *)
let data = Ndarray.of_array [|[|1.0; 2.0|]; [|3.0; 4.0|]; [|5.0; 6.0|]|] in
let label = Ndarray.of_array [|1; 0; 1|] in
let dataset = { data; label }  
(* Creating the dataset *)

(* Step 2: Shuffle the Dataset *)
let shuffled_dataset = shuffle dataset  
(* Shuffling the dataset to randomize the order *)

(* Step 3: Split the Dataset *)
let train_set, val_set = split dataset 0.8  
(* Splitting the dataset into 80% training and 20% validation *)

(* Step 4: Initialize the Dataloader *)
let batch_size = 2
let transfroms = [Transform.resize; Transform.flip]
let dataloader = Dataloader.create ~dataset:train_set ~batch_size ~shuffle:true ~transforms
(* Create the dataloader with batching *)

(* Step 5: Retrieve Batches *)
for i = 0 to Dataloader.get_total_batches dataloader - 1 do
  let batch_data, batch_label = Dataloader.get_batch dataloader i in
  (* Here, you can use batch_data and batch_label for model training or evaluation *)
done;
```

In this example:

1. **Dataset Creation**: We initialize a dataset with `data` and `label` values.
2. **Shuffle**: The dataset is shuffled to randomize the data order, which helps prevent the model from learning order-based patterns.
3. **Split**: The dataset is split into training and validation sets based on a specified ratio.
4. **Dataloader Initialization**: We create a `Dataloader` instance with the training set, specifying a `batch_size` and enabling shuffling.
5. **Batch Retrieval**: We loop through the batches, retrieving each batch's data and label for processing. This setup enables the model to train in mini-batches, a common practice in machine learning.



### Model Design

#### 1. Overview

##### Neural Network Layer Module

This module defines a structure for implementing neural network layers and modules. The primary components include parameterized layers (e.g., linear, convolutional) and activation functions (e.g., ReLU, Sigmoid), as well as essential loss functions like Mean Squared Error (MSE) and Cross Entropy.

##### Layer Structure

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

##### Key Layer Functions

The module provides essential functions for interacting with layers:

- **`forward`**: Executes a forward pass for a given layer with specified inputs.
- **`get_parameters`**: Retrieves the learnable parameters from a layer for use in optimization.
- **Layer Creators**: Factory functions for creating commonly used layers (e.g., Linear, Conv2d) and activation functions (e.g., ReLU, Sigmoid), as well as loss layers like MSE and Cross Entropy.

#### 2. Mock Use

The following example demonstrates how to create a simple neural network with a few layers, perform a forward pass, and retrieve the output.

##### Example Usage

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



### Optimizer Design

#### 1. Overview

##### Optimizer Module

The `Optimizer` module provides essential tools for updating model parameters during training by implementing commonly used optimizers like Stochastic Gradient Descent (SGD) and Adam.

##### Optimizer Structure

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

##### Supported Optimizers

1. **SGD (Stochastic Gradient Descent)**: Updates parameters by moving them in the direction of the gradient, scaled by a learning rate.
2. **Adam**: A more sophisticated optimizer that adapts the learning rate for each parameter based on moment estimates, enhancing convergence in many scenarios.

#### 2. Mock Use

Below is an example of how to use the `Optimizer` module to train a model's parameters.

##### Example Usage

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
utils.backprop loss;  (* Compute gradients via backpropagation *)

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
