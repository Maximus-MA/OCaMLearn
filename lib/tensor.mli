(* Type Definitions *)
type ndarray = {
  data: float array;   (* The tensor's data stored as a flat array *)
  shape: int array;    (* The tensor's shape as an array of dimensions *)
}

type t = {
  data: ndarray;                    (* The tensor's data as an ndarray *)
  grad: ndarray option;             (* Gradient of the tensor, if required *)
  requires_grad: bool;              (* Indicates if the tensor requires gradient computation *)
  backward_fn: (unit -> unit) option;  (* Function to compute gradients for backpropagation *)
  prev: t list;                     (* List of previous tensors for backpropagation *)
}

(* Creation functions *)
val create : data:float array -> shape:int array -> requires_grad:bool -> t
(** [create data shape requires_grad] creates a new tensor with specified [data], [shape], and [requires_grad] to indicate gradient requirement. *)

val from_tensor : t -> t
(** [from_tensor tensor] creates a copy of [tensor], inheriting its shape and gradient properties. *)

val zeros : int array -> t
(** [zeros shape] creates a tensor filled with zeros, of specified [shape]. *)

val ones : int array -> t
(** [ones shape] creates a tensor filled with ones, of specified [shape]. *)

val rand : int array -> t
(** [rand shape] creates a tensor with random values in the range [0,1] and the specified [shape]. *)

val xavier_init : int array -> t
(** [xavier_init shape] creates a tensor with Xavier initialization for the given [shape]. *)

val he_init : int array -> t
(** [he_init shape] creates a tensor with He initialization for the given [shape]. *)

(* Tensor information and properties *)
val ndim : t -> int
(** [ndim tensor] returns the number of dimensions of [tensor]. *)

val get_data : t -> float array
(** [get_data t] retrieves the raw data array of tensor [t]. *)

val shape : t -> int array
(** [shape t] returns the shape of tensor [t] as an array of dimensions. *)

val requires_grad : t -> bool
(** [requires_grad t] checks if tensor [t] requires gradient computation. *)

val get : t -> int array -> float
(** [get t idx] retrieves the value at the specified index [idx] in tensor [t]. *)

val set : t -> int array -> float -> unit
(** [set t idx value] sets the value at index [idx] in tensor [t] to [value]. *)

(* Gradient-related functions *)
val backward : t -> unit
(** [backward t] performs backpropagation on tensor [t] to compute gradients. *)

val zero_grad : t -> unit
(** [zero_grad t] resets the gradient of tensor [t] to zero. *)

val reset_grad : t -> unit
(** [reset_grad t] clears the gradient of tensor [t] but retains backpropagation history. *)

val accumulate_grad : t -> t -> unit
(** [accumulate_grad t grad] adds the given gradient tensor [grad] to the existing gradient of [t]. *)

val get_grad : t -> float array option
(** [get_grad t] retrieves the gradient of tensor [t] as an option type, if it exists. *)

val set_grad : t -> float array -> unit
(** [set_grad t grad] directly sets the gradient of tensor [t] to the given gradient array [grad]. *)

val clip_grad : t -> float -> unit
(** [clip_grad t max_val] clips the gradients of tensor [t] to the range [-max_val, max_val]. *)

(* Element-wise operations *)
val add : t -> t -> t
(** [add t1 t2] performs element-wise addition between tensors [t1] and [t2], supporting broadcasting. *)

val add_scalar : t -> float -> t
(** [add_scalar t x] adds scalar [x] to each element of tensor [t]. *)

val sub : t -> t -> t
(** [sub t1 t2] performs element-wise subtraction between tensors [t1] and [t2]. *)

val sub_scalar : t -> float -> t
(** [sub_scalar t x] subtracts scalar [x] from each element of tensor [t]. *)

val mul : t -> t -> t
(** [mul t1 t2] performs element-wise multiplication between tensors [t1] and [t2]. *)

val mul_scalar : t -> float -> t
(** [mul_scalar t x] multiplies each element in tensor [t] by scalar [x]. *)

val div : t -> t -> t
(** [div t1 t2] performs element-wise division of tensor [t1] by tensor [t2]. *)

val div_scalar : t -> float -> t
(** [div_scalar t x] divides each element in tensor [t] by scalar [x]. *)

(* Matrix operations *)
val matmul : t -> t -> t
(** [matmul t1 t2] performs matrix multiplication between tensors [t1] and [t2]. *)

val transpose : t -> t
(** [transpose t] returns the transpose of tensor [t]. *)

(* Tensor shape operations *)
val reshape : t -> shape:int array -> t
(** [reshape t ~shape] reshapes tensor [t] to the specified [shape]. *)

val expand : t -> int array -> t
(** [expand t new_shape] expands tensor [t] to match [new_shape] using broadcasting if necessary. *)

val concatenate : t list -> int -> t
(** [concatenate ts dim] concatenates a list of tensors [ts] along the specified dimension [dim]. *)

val split : t -> int -> t list
(** [split t dim] splits tensor [t] along the specified dimension [dim] into a list of sub-tensors. *)

(* Mathematical functions *)
val sum : t -> t
(** [sum t] calculates the sum of all elements in tensor [t]. *)

val mean : t -> t
(** [mean t] calculates the mean of all elements in tensor [t]. *)

val variance : t -> t
(** [variance t] calculates the variance of all elements in tensor [t]. *)

val std : t -> t
(** [std t] calculates the standard deviation of all elements in tensor [t]. *)

val normalize : t -> t
(** [normalize t] normalizes tensor [t] to have a mean of 0 and standard deviation of 1. *)

val max : t -> t
(** [max t] finds the maximum value in tensor [t]. *)

val min : t -> t
(** [min t] finds the minimum value in tensor [t]. *)

val argmax : t -> t
(** [argmax t] returns the indices of maximum values in tensor [t]. *)

val argmin : t -> t
(** [argmin t] returns the indices of minimum values in tensor [t]. *)

val dsum : t -> int -> t
(** [dsum t dim] calculates the sum of elements along the specified dimension [dim] in tensor [t]. *)

val dmean : t -> int -> t
(** [dmean t dim] calculates the mean of elements along the specified dimension [dim] in tensor [t]. *)

val dvariance : t -> int -> t
(** [dvariance t dim] calculates the variance of elements along the specified dimension [dim] in tensor [t]. *)

val dstd : t -> int -> t
(** [dstd t dim] calculates the standard deviation of elements along the specified dimension [dim] in tensor [t]. *)

val dnormalize : t -> int -> t
(** [dnormalize t dim] normalizes tensor [t] along dimension [dim] to have mean 0 and std 1. *)

val dmax : t -> int -> t
(** [dmax t dim] finds the maximum values along the specified dimension [dim] in tensor [t]. *)

val dmin : t -> int -> t
(** [dmin t dim] finds the minimum values along the specified dimension [dim] in tensor [t]. *)

val dargmax : t -> int -> t
(** [dargmax t dim] returns the indices of maximum values along the specified dimension [dim] in tensor [t]. *)

val dargmin : t -> int -> t
(** [dargmin t dim] returns the indices of minimum values along the specified dimension [dim] in tensor [t]. *)

(* Element-wise mathematical functions *)
val exp : t -> t
(** [exp t] computes the exponential of each element in tensor [t]. *)

val log : t -> t
(** [log t] computes the natural logarithm of each element in tensor [t]. *)

val pow : t -> int -> t
(** [pow t x] raises each element in tensor [t] to the power of [x]. *)

val sqrt : t -> t
(** [sqrt t] computes the square root of each element in tensor [t]. *)

(* Utility functions *)
val clone : t -> t
(** [clone t] creates a deep copy of tensor [t], including data and gradient. *)

val can_broadcast : t -> t -> bool
(** [can_broadcast t1 t2] checks if tensors [t1] and [t2] have compatible shapes for broadcasting. *)

val detach : t -> t
(** [detach t] returns a copy of tensor [t] without gradient tracking. *)
