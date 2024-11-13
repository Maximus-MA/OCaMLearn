(* Type Definitions *)
type ndarray = {
  data: float array;   (* The tensor's data stored as a flat array *)
  shape: int array;    (* The tensor's shape as an array of dimensions *)
}

type tensor = {
  data: ndarray;                    (* The actual tensor data *)
  grad: ndarray option;             (* Gradient of the tensor if required *)
  requires_grad: bool;              (* Indicates if the tensor requires gradient *)
  backward_fn: (unit -> unit) option;  (* Function to compute gradients for backpropagation *)
  prev: tensor list;                (* List of previous tensors for backpropagation *)
}

(* Creation functions *)
val create : data:float array -> shape:int array -> requires_grad:bool -> tensor
(** [create data shape requires_grad] creates a new tensor with given data, shape, and gradient requirement. *)

val from_tensor : ?requires_grad:bool -> tensor -> tensor
(** [from_tensor ?requires_grad tensor] creates a tensor from an existing tensor, with optional gradient requirement. *)

val zeros : ?requires_grad:bool -> int array -> tensor
(** [zeros ?requires_grad shape] creates a tensor filled with zeros, of specified shape and gradient option. *)

val ones : ?requires_grad:bool -> int array -> tensor
(** [ones ?requires_grad shape] creates a tensor filled with ones, of specified shape and gradient option. *)

val rand : ?requires_grad:bool -> int array -> tensor
(** [rand ?requires_grad shape] creates a tensor with random values, of specified shape and gradient option. *)

val xavier_init : ?requires_grad:bool -> int array -> tensor
(** [xavier_init ?requires_grad shape] creates a tensor with Xavier initialization. *)

val he_init : ?requires_grad:bool -> int array -> tensor
(** [he_init ?requires_grad shape] creates a tensor with He initialization. *)

(* Tensor information and properties *)
val ndim : tensor -> int
(** [ndim tensor] returns the number of dimensions of tensor. *)

val get_data : tensor -> float array
(** [get_data tensor] retrieves the data of tensor. *)

val shape : tensor -> int array
(** [shape tensor] returns the shape of tensor. *)

val requires_grad : tensor -> bool
(** [requires_grad tensor] checks if tensor requires gradient computation. *)

val get : tensor -> int array -> float
(** [get tensor idx] retrieves the value at index [idx] in tensor. *)

val set : tensor -> int array -> float -> unit
(** [set tensor idx value] sets the value at index [idx] in tensor to [value]. *)

(* Gradient-related functions *)
val backward : tensor -> unit
(** [backward tensor] performs backpropagation on tensor to compute gradients. *)

val zero_grad : tensor -> unit
(** [zero_grad tensor] resets the gradient of tensor to zero. *)

val reset_grad : tensor -> unit
(** [reset_grad tensor] clears the gradient but keeps backpropagation history. *)

val accumulate_grad : tensor -> tensor -> unit
(** [accumulate_grad tensor grad] accumulates the given [grad] into tensor's existing gradient. *)

val get_grad : tensor -> float array option
(** [get_grad tensor] retrieves the gradient of tensor, if it exists. *)

val set_grad : tensor -> float array -> unit
(** [set_grad tensor grad] directly sets the gradient of tensor to [grad]. *)

val clip_grad : tensor -> float -> unit
(** [clip_grad tensor max_val] clips the gradients of tensor to the range [-max_val, max_val]. *)

(* Element-wise operations *)
val add : tensor -> tensor -> tensor
(** [add tensor1 tensor2] performs element-wise addition of tensors [tensor1] and [tensor2], supporting broadcasting. *)

val add_scalar : tensor -> float -> tensor
(** [add_scalar tensor x] adds scalar [x] to each element in tensor. *)

val sub : tensor -> tensor -> tensor
(** [sub tensor1 tensor2] performs element-wise subtraction between tensors [tensor1] and [tensor2]. *)

val sub_scalar : tensor -> float -> tensor
(** [sub_scalar tensor x] subtracts scalar [x] from each element in tensor. *)

val mul : tensor -> tensor -> tensor
(** [mul tensor1 tensor2] performs element-wise multiplication of tensors [tensor1] and [tensor2]. *)

val mul_scalar : tensor -> float -> tensor
(** [mul_scalar tensor x] multiplies each element in tensor by scalar [x]. *)

val div : tensor -> tensor -> tensor
(** [div tensor1 tensor2] performs element-wise division of tensor1 by tensor2. *)

val div_scalar : tensor -> float -> tensor
(** [div_scalar tensor x] divides each element in tensor by scalar [x]. *)

(* Matrix operations *)
val matmul : tensor -> tensor -> tensor
(** [matmul tensor1 tensor2] performs matrix multiplication between tensors [tensor1] and [tensor2]. *)

val transpose : tensor -> tensor
(** [transpose tensor] returns the transpose of tensor. *)

(* Tensor shape operations *)
val reshape : tensor -> shape:int array -> tensor
(** [reshape tensor ~shape] reshapes tensor to the specified shape. *)

val expand : tensor -> int array -> tensor
(** [expand tensor new_shape] expands tensor to match [new_shape] using broadcasting. *)

val concatenate : tensor list -> int -> tensor
(** [concatenate tensors dim] concatenates a list of tensors along the specified dimension. *)

val split : tensor -> int -> tensor list
(** [split tensor dim] splits tensor along the specified dimension into sub-tensors. *)

(* Mathematical functions *)
val sum : tensor -> ?dim:int -> tensor
(** [sum tensor ?dim] calculates the sum of all elements, or along the specified dimension. *)

val mean : tensor -> ?dim:int -> tensor
(** [mean tensor ?dim] calculates the mean of all elements, or along the specified dimension. *)

val variance : tensor -> ?dim:int -> tensor
(** [variance tensor ?dim] calculates the variance of all elements, or along the specified dimension. *)

val std : tensor -> ?dim:int -> tensor
(** [std tensor ?dim] calculates the standard deviation of all elements, or along the specified dimension. *)

val normalize : tensor -> ?dim:int -> tensor
(** [normalize tensor ?dim] normalizes tensor to mean 0 and std 1 along the specified dimension. *)

val max : tensor -> ?dim:int -> tensor
(** [max tensor ?dim] finds the maximum value in tensor, or along the specified dimension. *)

val min : tensor -> ?dim:int -> tensor
(** [min tensor ?dim] finds the minimum value in tensor, or along the specified dimension. *)

val argmax : tensor -> ?dim:int -> tensor
(** [argmax tensor ?dim] returns indices of max values along the specified dimension. *)

val argmin : tensor -> ?dim:int -> tensor
(** [argmin tensor ?dim] returns indices of min values along the specified dimension. *)

(* Element-wise mathematical functions *)
val exp : tensor -> tensor
(** [exp tensor] calculates the exponential of each element in tensor. *)

val log : tensor -> tensor
(** [log tensor] calculates the natural log of each element in tensor. *)

val pow : tensor -> int -> tensor
(** [pow tensor x] raises each element in tensor to the power of [x]. *)

val sqrt : tensor -> tensor
(** [sqrt tensor] calculates the square root of each element in tensor. *)

(* Utility functions *)
val clone : tensor -> tensor
(** [clone tensor] creates a deep copy of tensor, including data and gradient. *)

val can_broadcast : tensor -> tensor -> bool
(** [can_broadcast tensor1 tensor2] checks if tensors [tensor1] and [tensor2] have compatible shapes for broadcasting. *)

val detach : tensor -> tensor
(** [detach tensor] returns a copy of tensor without gradient tracking. *)
