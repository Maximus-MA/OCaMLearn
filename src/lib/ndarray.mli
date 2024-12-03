type t = {
  data: float array;  (* The underlying data array of the ndarray. *)
  shape: int array;   (* The shape of the ndarray. *)
}

(* Basic Operations *)
val numel : int array -> int
(** [numel shape] computes the total number of elements for the given [shape]. *)

val strides : int array -> int array
(** [strides shape] calculates the strides for an ndarray with the specified [shape]. *)

val sum_multiplied : float array -> float array -> float
(** [sum_multiplied a b] computes the sum of element-wise products of arrays [a] and [b]. *)

(* Broadcasting Utilities *)
val is_broadcastable : int array -> int array -> bool
(** [is_broadcastable shape1 shape2] checks if two shapes are compatible for broadcasting. *)

val broadcast_shape : int array -> int array -> int array
(** [broadcast_shape shape1 shape2] returns the shape resulting from broadcasting [shape1] and [shape2]. *)

(* Element-wise Arithmetic Operations *)
val add : t -> t -> t
(** [add a b] performs element-wise addition with broadcasting. *)

val sub : t -> t -> t
(** [sub a b] performs element-wise subtraction with broadcasting. *)

val mul : t -> t -> t
(** [mul a b] performs element-wise multiplication with broadcasting. *)

val div : t -> t -> t
(** [div a b] performs element-wise division with broadcasting. *)

val matmul : t -> t -> t
(** [matmul a b] computes the matrix multiplication of [a] and [b], supporting batch dimensions. *)

(* Creation Functions *)
val create : float array -> int array -> t
(** [create data shape] initializes a new ndarray with the given [data] and [shape]. *)

val create_float : float -> t
(** [create_float value] creates an ndarray containing a single float [value]. *)

val create_int : int -> t
(** [create_int value] creates an ndarray containing a single integer [value]. *)

val zeros : int array -> t
(** [zeros shape] creates an ndarray of the specified [shape], filled with zeros. *)

val ones : int array -> t
(** [ones shape] creates an ndarray of the specified [shape], filled with ones. *)

val rand : int array -> t
(** [rand shape] generates an ndarray with random values between 0 and 1. *)

val xavier_init : int array -> t
(** [xavier_init shape] initializes an ndarray using Xavier initialization. *)

val kaiming_init : int array -> t
(** [kaiming_init shape] initializes an ndarray using Kaiming initialization. *)

(* Shape and Dimension Operations *)
val shape : t -> int array
(** [shape arr] returns the shape of the ndarray [arr]. *)

val dim : t -> int
(** [dim arr] retrieves the number of dimensions of the ndarray [arr]. *)

val at : t -> int array -> float
(** [at arr indices] retrieves the value at the specified [indices] in [arr]. *)

val reshape : t -> int array -> t
(** [reshape arr new_shape] reshapes the ndarray [arr] to [new_shape]. *)

val set : t -> int array -> float -> unit
(** [set arr idx value] updates the value at index [idx] in [arr] to [value]. *)

val transpose : t -> t
(** [transpose arr] returns the transpose of the ndarray [arr]. *)

val to_array : t -> float array
(** [to_array arr] converts the ndarray [arr] back to a flat float array. *)

val slice : t -> (int * int) list -> t
(** [slice arr ranges] extracts a sub-array from [arr] based on the specified [ranges]. *)

(* Modifications and Aggregations *)
val fill : t -> float -> unit
(** [fill arr value] sets all elements in [arr] to [value]. *)

val sum : t -> float
(** [sum arr] computes the sum of all elements in [arr]. *)

val mean : t -> float
(** [mean arr] calculates the mean of all elements in [arr]. *)

val var : t -> float
(** [var arr] calculates the variance of all elements in [arr]. *)

val std : t -> float
(** [std arr] calculates the standard deviation of all elements in [arr]. *)

val max : t -> float
(** [max arr] finds the maximum value in [arr]. *)

val min : t -> float
(** [min arr] finds the minimum value in [arr]. *)

val argmax : t -> int
(** [argmax arr] returns the index of the maximum value in [arr]. *)

val argmin : t -> int
(** [argmin arr] returns the index of the minimum value in [arr]. *)

(* Dimensional Reductions *)
val dsum : t -> int -> t
(** [dsum arr dim] computes the sum along the specified dimension [dim]. *)

val dmean : t -> int -> t
(** [dmean arr dim] computes the mean along the specified dimension [dim]. *)

val dvar : t -> int -> t
(** [dvar arr dim] calculates the variance along the specified dimension [dim]. *)

val dstd : t -> int -> t
(** [dstd arr dim] calculates the standard deviation along the specified dimension [dim]. *)

val dmax : t -> int -> t
(** [dmax arr dim] finds the maximum value along the specified dimension [dim]. *)

val dmin : t -> int -> t
(** [dmin arr dim] finds the minimum value along the specified dimension [dim]. *)

val dargmax : t -> int -> t
(** [dargmax arr dim] returns the indices of maximum values along [dim]. *)

val dargmin : t -> int -> t
(** [dargmin arr dim] returns the indices of minimum values along [dim]. *)

(* Element-wise Mathematical Operations *)
val exp : t -> t
(** [exp arr] computes the exponential of each element in [arr]. *)

val log : t -> t
(** [log arr] computes the natural logarithm of each element in [arr]. *)

val sqrt : t -> t
(** [sqrt arr] computes the square root of each element in [arr]. *)

val pow : t -> float -> t
(** [pow arr x] raises each element in [arr] to the power of [x]. *)

(* Dimension Manipulations *)
val expand_dims : t -> int -> t
(** [expand_dims arr dim] inserts a new dimension of size 1 at the specified [dim]. *)

val squeeze : t -> t
(** [squeeze arr] removes dimensions of size 1 from [arr]. *)


val map : t-> f: (float -> float) ->t

val reduce_sum_to_shape : t -> int array -> t

val negate: t->t

val relu: t->t
