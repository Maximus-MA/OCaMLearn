# Tensor Design



## 1. Overview

### `Tensor` Module

The `Tensor` module builds on top of `ndarray`, enabling gradient tracking and backpropagation capabilities essential for machine learning.

```ocaml
type ndarray = Ndarray.t

type t = {
  data: ndarray;                    (* The tensor's data as an ndarray *)
  grad: ndarray option;             (* Gradient of the tensor, if required *)
  requires_grad: bool;              (* Indicates if the tensor requires gradient computation *)
  backward_fn: (unit -> unit) option;  
  (* Function to compute gradients for backpropagation *)
  prev: t list;                     
  (* List of previous tensors for backpropagation *)
}
```



In the `Tensor` structure:

- **`data`**: Stores the actual tensor data as an `ndarray`.
- **`grad`**: Stores the gradient of `data`, which can be computed during backpropagation if `requires_grad` is `true`.
- **`requires_grad`**: Indicates whether gradient computation is necessary. If `true`, gradients are stored in `grad`.
- **`backward_fn`**: Defines the backpropagation function for the tensor, specifying how gradients should be computed.
- **`prev`**: Stores a list of previous tensors, allowing construction of the computation graph for backpropagation.

The `Tensor` module allows for gradient-based optimization by constructing computation graphs and enabling automatic differentiation, similar to the functionality in Python’s PyTorch library.



## 2. Mock Use

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



## 3. Library Dependencies

We primarily use standard OCaml libraries:

- **Array**: For standard array manipulations.
- **Stdlib**: For basic OCaml functions and utilities.

These libraries are part of the OCaml standard library, so no additional testing is required for them.



## 4. Implementation Plan

Our implementation plan is as follows:

1. **This Week**: **Chuan Chen** and **Rui Wang** aim to implement the basic functionality of the `Tensor` module, including element-wise addition, subtraction, multiplication, and division operations with broadcasting support. During these operations, the newly generated tensor will record its predecessors, enabling the construction of a computational graph.
2. **Next Week**: **Chuan Chen** and **Rui Wang** will implement the backward propagation functions for each operation. This will allow us to support automatic differentiation by propagating gradients through the computational graph, thereby completing the core functionality of the `Tensor` module.



