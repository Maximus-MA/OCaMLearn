type tensor = tensor.t
(** An alias for the tensor type from the tensor module, representing the data structure for tensor data and gradients. *)

type t = {
    data: tensor;
    grad: tensor;
    requires_grad: bool;
    backward_fn: (unit -> unit) option;
    prev: t list;
}
(** A type [t] representing a tensor, with:
    - [data]: the tensor containing the tensor's data.
    - [grad]: an tensor holding the tensor's gradient.
    - [requires_grad]: a boolean indicating if gradient tracking is enabled.
    - [backward_fn]: an optional function to compute gradients for backpropagation.
    - [prev]: a list of preceding tensors, representing dependencies for gradient computation.
*)

val from_tensor : ?requires_grad:bool -> tensor -> t
(** [from_tensor ?requires_grad tensor] creates a tensor from the given [tensor], with optional [requires_grad] to enable gradient computation. *)

val zeros : ?requires_grad:bool -> int array -> t
(** [zeros ?requires_grad shape] creates a tensor filled with zeros, with the specified [shape] and an optional [requires_grad] flag. *)

val ones : ?requires_grad:bool -> int array -> t
(** [ones ?requires_grad shape] creates a tensor filled with ones, with the specified [shape] and an optional [requires_grad] flag. *)

val rand : ?requires_grad:bool -> int array -> t
(** [rand ?requires_grad shape] creates a tensor with random values, of the specified [shape] and an optional [requires_grad] flag. *)

val ndim : t -> int
(** [ndim t] returns the number of dimensions of tensor [t]. *)

val get : t -> int array -> float
(** [get t index_list] returns the value of the tensor [t] at the specified index list[index_list]. *)

val set : t -> int array -> float -> unit
(** [set t idx value] updates the element at index [idx] in [t] with [value]. *)

val to_tensor : t -> tensor
(** [to_tensor t] returns the underlying tensor data of tensor [t]. *)

val get_grad : t -> tensor
(** [get_grad t] returns the gradient of tensor [t] as an tensor. *)

val set_grad : t -> tensor -> unit
(** [set_grad t grad] sets the gradient of tensor [t] in-place to the given [grad] value.
    This updates the tensor's gradient directly, which is more efficient for mutable arrays.
*)

val get_requires_grad : t -> bool
(** [get_requires_grad t] checks whether tensor [t] requires gradient computation. *)

val set_requires_grad : t -> bool -> t
(** [set_requires_grad t flag] sets the [requires_grad] attribute of tensor [t] to [flag], returning a new tensor if [flag] changes. *)

val zero_grad : t -> unit
(** [zero_grad t] resets the gradient of tensor [t] to zero. *)

val add : t -> t -> t
(** [add t1 t2] performs element-wise addition of tensors [t1] and [t2], supporting broadcasting. *)

val add_scalar : t -> float -> t
(** [add_scalar t x] adds scalar [x] to each element of tensor [t]. *)

val sub : t -> t -> t
(** [sub t1 t2] performs element-wise subtraction of tensor [t2] from [t1], supporting broadcasting. *)

val sub_scalar : t -> float -> t
(** [sub_scalar t x] subtracts scalar [x] from each element of tensor [t]. *)

val mul : t -> t -> t
(** [mul t1 t2] performs element-wise multiplication of tensors [t1] and [t2], supporting broadcasting. *)

val mul_scalar : t -> float -> t
(** [mul_scalar t x] multiplies each element of tensor [t] by scalar [x]. *)

val div : t -> t -> t
(** [div t1 t2] performs element-wise division of tensor [t1] by tensor [t2], supporting broadcasting. *)

val div_scalar : t -> float -> t
(** [div_scalar t x] divides each element of tensor [t] by scalar [x]. *)

val matmul : t -> t -> t
(** [matmul t1 t2] performs matrix multiplication between tensors [t1] and [t2]. *)

val transpose : t -> t
(** [transpose t] returns the transpose of tensor [t]. *)

val reshape : t -> shape:int array -> t
(** [reshape t ~shape] reshapes tensor [t] to the specified [shape]. *)

val sum : t -> ?dim:int -> t
(** [sum ?dim t] computes the sum of all elements in tensor [t]. 
    If [dim] is provided, it computes the sum along the specified axes, 
    returning a tensor with those dimensions reduced. *)

val mean : t -> ?dim:int -> t
(** [mean ?dim t] computes the mean of all elements in tensor [t]. 
    If [dim] is provided, it computes the mean along the specified axes, 
    returning a tensor with those dimensions reduced. *)

val max : t -> ?dim:int -> t
(** [max ?dim t] computes the maximum value of all elements in tensor [t]. 
    If [dim] is provided, it computes the maximum along the specified axes, 
    returning a tensor with those dimensions reduced. *)

val min : t -> ?dim:int -> t
(** [min ?dim t] computes the minimum value of all elements in tensor [t]. 
    If [dim] is provided, it computes the minimum along the specified axes, 
    returning a tensor with those dimensions reduced. *)

val argmax : t -> ?dim:int -> t
(** [argmax ?dim t] returns a tensor with the indices of the maximum values along the specified [dim] of tensor [t].
    If [dim] is not provided, it returns the index of the maximum value in the flattened tensor.
*)

val argmin : t -> ?dim:int -> t
(** [argmin ?dim t] returns a tensor with the indices of the minimum values along the specified [dim] of tensor [t].
    If [dim] is not provided, it returns the index of the minimum value in the flattened tensor.
*)

val variance : t -> ?dim:int -> t
(** [variance t] calculates the variance of all elements in the tensor [t]. 
    If [dim] is provided, it computes the variance along the specified dim, 
    returning a tensor with those dimensions reduced. *)

val std : t -> ?dim:int -> t
(** [std t] calculates the standard deviation of all elements in the tensor [t]. 
    If [dim] is provided, it computes the std along the specified dim, 
    returning a tensor with those dimensions reduced. *)

val normalize : t -> ?dim:int -> t
(** [normalize t] normalizes tensor [t] by scaling its values to have a mean of 0 and a standard deviation of 1.
    If [dim] is provided, it normalizes along the specified dim, 
    returning a tensor with those dimensions reduced. *)    

val exp : t -> t
(** [exp t] computes the exponential of each element in tensor [t]. *)

val log : t -> t
(** [log t] computes the natural logarithm of each element in tensor [t]. *)

val pow : t -> int -> t
(** [pow t x] raises each element in tensor [t] to the power [x]. *)

val sqrt : t -> t
(** [sqrt t] computes the square root of each element in tensor [t]. *)

val clone : t -> t
(** [clone t] creates a deep copy of tensor [t], including its data and gradient properties. *)

val concatenate : t list -> int -> t
(** [concatenate tensors dim] concatenates the list of tensors [tensors] along the specified [dim]. *)

val split : t -> int -> t list
(** [split t dim] splits tensor [t] along the specified [dim] into a list of sub-tensors. *)

val detach : t -> t
(** [detach t] creates a copy of tensor [t] with no gradient tracking. 
    This is useful to stop a tensor from participating in gradient computation. *)
