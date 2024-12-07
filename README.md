# Neural Network Library in OCaml

## Overview

Our goal is to implement a basic neural network library in OCaml, inspired by PyTorch. The library provides fundamental data structures and operations for building and training neural networks, including:

- An `ndarray` module that handles multi-dimensional arrays and their common operations.
- A `tensor` module built on top of `ndarray` that adds features for automatic differentiation (backpropagation).

By the end of this project, we aim to have a working system where one can define computation graphs and train simple models. This initial checkpoint demonstrates our current progress, module structure, and approach to testing and coverage.



## Progress So Far

### `ndarray` Module

**Status:** Complete.

- **Features Implemented:**
  - Basic data structure for multi-dimensional arrays (`float array` for data and `int array` for shape).
  - Core operations such as addition, subtraction, multiplication, division, and broadcasting logic.
  - Linear algebra operations like `matmul`.
  - Reduction functions (`sum`, `mean`, `var`, `std`, `max`, `min`).
  - Utility functions for slicing, reshaping, indexing, and element-wise transformations (`exp`, `log`, `pow`).
  - Random initialization methods (`rand`, `xavier_init`, `kaiming_init`).
  - Broadcasting and shape manipulation logic.
- **Testing:**
  - We have tests that cover:
    - Basic arithmetic operations on `ndarray`.
    - Broadcasting behavior.
    - Shape transformations (reshape, slice).
    - Reduction operations (sum, mean).
  - Tests are executed with `dune test`, and we have measured coverage to ensure a good portion of the `ndarray` module is tested.

**What’s Working:**
The `ndarray` module should be stable and fully functional for standard array operations needed in neural network computations.

**What’s Not Working / Next Steps:**
Some more complex linear algebra operations and advanced indexing are not fully explored. Additional edge cases may also need to be tested.



### `Tensor` Module

**Status:** In progress.

- **Features Implemented:**
  - A `tensor` type wrapping an `ndarray` and providing `grad` fields for gradients.
  - Basic forward operations (`add`, `sub`, `mul`, `div`, `matmul`) and their backward passes.
  - Simple activation function (`relu`).
  - Partial support for functions like `sum` and `mean` with backpropagation.
- **Automatic Differentiation:**
  - We have implemented a `backward_fn` mechanism that stores references to predecessor tensors and calculates gradients when `backward()` is triggered.
  - The current code supports gradient propagation for a subset of operations.
- **Testing:**
  - We have preliminary tests that:
    - Create tensors and run forward operations.
    - Execute backward calls to check if gradients are computed correctly for simple arithmetic and `relu`.
  - Coverage reports show that a subset of `tensor` functionality is tested. We will increase coverage as more features are added.

**What’s Working:**
Most basic arithmetic operations and `relu` can do both forward and backward passes. We can verify correctness through a small demo where a simple training loop shows a loss function converging.

**What’s Not Working / Next Steps:**

- Many higher-level operations (`log`, `exp`, `softmax`, `log_softmax`, etc.) are currently placeholders or partially implemented, especially their gradients.
- Need to improve gradient broadcasting and ensure all backprop logic is correct.
- Additional testing and coverage for backward passes and complex operations are needed.
- Implement more advanced neural network layers and a flexible graph structure.



### Demo

**Status:** Basic demo completed.

We have a simple demo training loop where a small model (e.g., a linear model) is trained on a dummy dataset. The loss decreases over training iterations, providing a preliminary validation of the correctness of forward and backward computations.



## Code Coverage

We wrote the tests for ndarray and tensor in **test/**, and we had to change the code in dune to be able to test ndarray and tensor in turn. That is, we have to comment one of them to be able to test it.

**Dune for testing ndarray:**

```ocaml
(test
 (name ndarray_tests)
 (libraries ounit2 ndarray)
 (preprocess (pps bisect_ppx)));

 ; (test
 ; (name tensor_tests)
 ; (libraries ounit2 ndarray tensor)
 ; (preprocess (pps bisect_ppx)));
```

**Dune for testing tensor:**

```ocaml
;(test
; (name ndarray_tests)
; (libraries ounit2 ndarray)
; (preprocess (pps bisect_ppx)));

(test
(name tensor_tests)
(libraries ounit2 ndarray tensor)
(preprocess (pps bisect_ppx)));
```





## Next Steps Before Final Submission

- Complete implementation of more element-wise operations and their gradients (`exp`, `log`, `sqrt`, `pow`, `softmax`, `log_softmax`).
- Add more complex layers and a neural network training pipeline.
- Refine gradient checks and implement utilities for gradient clipping, normalization, etc.
- Improve test coverage and handle edge cases.
- Ensure all code that is not working is either fixed or commented out.



## Conclusion

At this code checkpoint, we have approximately half of the core functionality in place. The `ndarray` module is nearly complete and well-tested, while the `tensor` module implements a good subset of the planned automatic differentiation features. With a solid foundation in place, we will spend the remaining time finalizing operations, improving the backward passes, enhancing test coverage, and implementing a small but functional neural network training example (MNIST).

## High-level Example
Here is an example to use our framework to train a MLP on random data.
```ocaml
let input = Tensor.ones [|2; 100|] in
  let model = Model.create_Sequential [
    Model.create_Linear ~in_features:100 ~out_features:50 ~bias:true;
    Model.create_ReLU ();
    Model.create_Linear ~in_features:50 ~out_features:10 ~bias:true;
    ] in 
  let loss_func = Model.create_CrossEntropy () in
  let target =  Tensor.zeros [|2; 10|] in
  Tensor.set target [|0; 0|] 1.;
  Tensor.set target [|1; 2|] 1.;
  let optimizer = Optimizer.create_SGD ~params:Model.(model.parameters) ~lr:0.01 in
  for _ = 0 to 10 do
    let output = Model.forward model [input] in
    let loss = Model.forward loss_func [output; target] in
    Printf.printf "Loss: %s\n" (Tensor.to_string loss);
    optimizer.zero_grad ();
    Utils.backprop loss;
    optimizer.step ()
  done
```
output
```
Loss: Tensor {data = 38.235257119, requires_grad = true}
Loss: Tensor {data = 32.6014171402, requires_grad = true}
Loss: Tensor {data = 27.6126259998, requires_grad = true}
Loss: Tensor {data = 22.8692291298, requires_grad = true}
Loss: Tensor {data = 18.2847880817, requires_grad = true}
Loss: Tensor {data = 13.7752143721, requires_grad = true}
Loss: Tensor {data = 9.25994272481, requires_grad = true}
Loss: Tensor {data = 4.88796455224, requires_grad = true}
Loss: Tensor {data = 3.04297275696, requires_grad = true}
Loss: Tensor {data = 0.733882587709, requires_grad = true}
Loss: Tensor {data = 0.836976953813, requires_grad = true}
```
According to output, we can see that the model is converging!
