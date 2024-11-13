(* src/ndarray.mli *)

type t = {
  data: float array;
  shape: int array;
}

val numel : int array -> int
(** [numel shape] returns the total number of elements for an ndarray with the given [shape]. *)

val strides : int array -> int array
(** [strides shape] returns the strides for an ndarray with the given [shape]. *)

val sum_multiplied : float array -> float array -> float
(** [sum_multiplied a b] computes the sum of element-wise products of arrays [a] and [b]. *)

val is_broadcastable : int array -> int array -> bool
(** [is_broadcastable shape1 shape2] checks if two shapes [shape1] and [shape2] are compatible for broadcasting. *)

val broadcast_shape : int array -> int array -> int array
(** [broadcast_shape shape1 shape2] returns the resulting shape when broadcasting [shape1] and [shape2]. *)

val add : t -> t -> t
(** [add a b] performs element-wise addition of ndarrays [a] and [b] with broadcasting. *)

val sub : t -> t -> t
(** [sub a b] performs element-wise subtraction of ndarrays [a] and [b] with broadcasting. *)

val mul : t -> t -> t
(** [mul a b] performs element-wise multiplication of ndarrays [a] and [b] with broadcasting. *)

val div : t -> t -> t
(** [div a b] performs element-wise division of ndarrays [a] and [b] with broadcasting. *)

val matmul : t -> t -> t
(** [matmul a b] performs matrix multiplication on ndarrays [a] and [b], supporting batch dimensions. *)

val create : float array -> int array -> t
(** [create data shape] creates a new ndarray with the specified [data] and [shape]. *)

val create_float : float -> t
(** [create_float value] creates a new ndarray initialized with a single float [value]. *)

val create_int : int -> t
(** [create_int value] creates a new ndarray initialized with a single integer [value]. *)

val zeros : int array -> t
(** [zeros shape] creates an ndarray of the given [shape] filled with zeros. *)

val ones : int array -> t
(** [ones shape] creates an ndarray of the given [shape] filled with ones. *)

val rand : int array -> t
(** [rand shape] creates an ndarray of the given [shape] filled with random values between 0 and 1. *)

val xavier_init : int array -> t
(** [xavier_init shape] creates an ndarray with the given [shape] initialized using Xavier initialization. *)

val kaiming_init : int array -> t
(** [kaiming_init shape] creates an ndarray with the given [shape] initialized using Kaiming initialization. *)

val shape : t -> int array
(** [shape arr] returns the shape of the ndarray [arr]. *)

val dim : t -> int
(** [dim arr] returns the number of dimensions of the ndarray [arr]. *)

val at : t -> int array -> float
(** [at arr indices] returns the value at the specified [indices] in the ndarray [arr]. *)

val reshape : t -> int array -> t
(** [reshape arr new_shape] reshapes the ndarray [arr] to [new_shape] and returns a new ndarray. *)

val set : t -> int array -> float -> unit
(** [set arr idx value] updates the element at index [idx] in [arr] with [value]. *)

val transpose : t -> t
(** [transpose arr] returns the transpose of the ndarray [arr]. *)

val to_array : t -> float array
(** [to_array arr] converts the ndarray [arr] to a float array. *)

val slice : t -> (int * int) list -> t
(** [slice arr ranges] extracts a sub-array from [arr] based on [ranges], where each pair in [ranges] specifies the start and end index for each dimension. *)

val fill : t -> float -> unit
(** [fill arr value] sets all elements in [arr] to the specified [value]. *)

val sum : t -> float
(** [sum arr] computes the sum of all elements in the ndarray [arr]. *)

val mean : t -> float
(** [mean arr] computes the mean of all elements in the ndarray [arr]. *)

val var : t -> float
(** [var arr] calculates the variance of all elements in the ndarray [arr]. *)

val std : t -> float
(** [std arr] calculates the standard deviation of all elements in the ndarray [arr]. *)

val max : t -> float
(** [max arr] computes the maximum value of all elements in the ndarray [arr]. *)

val min : t -> float
(** [min arr] computes the minimum value of all elements in the ndarray [arr]. *)

val argmax : t -> int
(** [argmax arr] returns the index of the maximum value in the ndarray [arr]. *)

val argmin : t -> int
(** [argmin arr] returns the index of the minimum value in the ndarray [arr]. *)

val dsum : t -> int -> t
(** [dsum arr dim] computes the sum of elements in [arr] along the specified dimension [dim], returning an ndarray with that dimension reduced. *)

val dmean : t -> int -> t
(** [dmean arr dim] computes the mean of elements in [arr] along the specified dimension [dim], returning an ndarray with that dimension reduced. *)

val dvar : t -> int -> t
(** [dvar arr dim] calculates the variance of elements in [arr] along the specified dimension [dim], returning an ndarray with that dimension reduced. *)

val dstd : t -> int -> t
(** [dstd arr dim] calculates the standard deviation of elements in [arr] along the specified dimension [dim], returning an ndarray with that dimension reduced. *)

val dmax : t -> int -> t
(** [dmax arr dim] computes the maximum value of elements in [arr] along the specified dimension [dim], returning an ndarray with that dimension reduced. *)

val dmin : t -> int -> t
(** [dmin arr dim] computes the minimum value of elements in [arr] along the specified dimension [dim], returning an ndarray with that dimension reduced. *)

val dargmax : t -> int -> t
(** [dargmax arr dim] returns an ndarray with the indices of the maximum values along the specified dimension [dim] of [arr]. *)

val dargmin : t -> int -> t
(** [dargmin arr dim] returns an ndarray with the indices of the minimum values along the specified dimension [dim] of [arr]. *)

val exp : t -> t
(** [exp arr] computes the exponential of each element in [arr]. *)

val log : t -> t
(** [log arr] computes the natural logarithm of each element in [arr]. *)

val sqrt : t -> t
(** [sqrt arr] computes the square root of each element in [arr]. *)

val pow : t -> float -> t
(** [pow arr x] raises each element in [arr] to the power of [x]. *)

val expand_dims : t -> int -> t
(** [expand_dims arr dim] adds a new dimension of size 1 at the specified [dim] in [arr]. *)

val squeeze : t -> t
(** [squeeze arr] removes dimensions of size 1 from [arr]. *)
