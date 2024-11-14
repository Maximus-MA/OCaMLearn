# Ndarray Design



## 1. Overview

Our project is designed with `ndarray` and `tensor` as the fundamental computational modules.

### `Ndarray` Module

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



## 2. Mock Use

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



## 3. Library Usage

We used the following libraries:

- **Array**: Standard OCaml array operations.
- **Stdlib**: OCaml standard library for fundamental data structures and utilities.

Since these are standard libraries, no additional testing was required for these components.



## 4. Implementation Plan

We have implemented almost every function for `ndarray` and written thorough tests for all finished functions. This includes:

- Initialization methods (custom, Xavier, Kaiming)
- Broadcasting operations
- Element-wise and matrix operations
- Statistical functions (e.g., `sum`, `mean`, `max`, `min`) with axis support
- Reshaping and slicing functionalities

Our tests validate that each function performs as expected, ensuring robust and reliable operation of the library.
