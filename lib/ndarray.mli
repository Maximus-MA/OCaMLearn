(* src/ndarray.mli *)

type ndarray = {
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

val add : ndarray -> ndarray -> ndarray
(** [add a b] performs element-wise addition of ndarrays [a] and [b] with broadcasting. *)

val sub : ndarray -> ndarray -> ndarray
(** [sub a b] performs element-wise subtraction of ndarrays [a] and [b] with broadcasting. *)

val mul : ndarray -> ndarray -> ndarray
(** [mul a b] performs element-wise multiplication of ndarrays [a] and [b] with broadcasting. *)

val div : ndarray -> ndarray -> ndarray
(** [div a b] performs element-wise division of ndarrays [a] and [b] with broadcasting. *)

val matmul : ndarray -> ndarray -> ndarray
(** [matmul a b] performs matrix multiplication on ndarrays [a] and [b], including batch dimensions. *)

val create : float array -> int array -> ndarray
(** [create data shape] creates a new ndarray with the given [data] and [shape]. *)

val create_float : float -> ndarray
(** [create data] creates a new ndarray with the one given [data]. *)

val create_int : int -> ndarray
(** [create data] creates a new ndarray with the one given [data]. *)

val zeros : int array -> ndarray
(** [zeros shape] creates an ndarray of the given [shape] filled with zeros. *)

val ones : int array -> ndarray
(** [ones shape] creates an ndarray of the given [shape] filled with ones. *)

val rand : int array -> ndarray
(** [rand shape] creates an ndarray of the given [shape] filled with random values between 0 and 1. *)

val xavier_init : int array -> ndarray
(** [xavier_init shape] creates an ndarray with the given [shape] initialized using Xavier initialization. *)

val kaiming_init : int array -> ndarray
(** [kaiming_init shape] creates an ndarray with the given [shape] initialized using Kaiming initialization. *)

val shape : ndarray -> int array
(** [shape arr] returns the shape of the ndarray [arr]. *)

val dim : ndarray -> int
(** [dim arr] returns the number of dimensions of the ndarray [arr]. *)

val at : ndarray -> int array -> float
(** [at arr indices] returns the value at the specified [indices] in the ndarray [arr]. *)

val reshape : ndarray -> int array -> ndarray
(** [reshape arr new_shape] reshapes the ndarray [arr] to [new_shape] and returns a new ndarray. *)

val set : ndarray -> int array -> float -> unit
(** [set t idx value] updates the element at index [idx] in [t] with [value]. *)

val transpose : ndarray -> ndarray
(** [transpose t] returns the transpose of ndarray [t]. *)

val to_array : ndarray -> float array
(** [to_array t] converts the ndarray [t] to a float array. *)

val slice : ndarray -> (int * int) list -> ndarray
(** [slice t ranges] extracts a sub-array from [t] based on [ranges], where each pair in [ranges] specifies the start and end index for each dimension. *)

val fill : ndarray -> float -> unit
(** [fill t value] sets all elements in [t] to the specified [value]. *)

val sum : ndarray -> float
(** [sum ?dim t] computes the sum of all elements in ndarray [t]. *)

val mean : ndarray -> float
(** [mean ?dim t] computes the mean of all elements in ndarray [t]. *)

val var : ndarray -> float
(** [var t] calculates the variance of all elements in the ndarray [t]. *)

val std : ndarray -> float
(** [std t] calculates the standard deviation of all elements in the ndarray [t]. *)

val max : ndarray -> float
(** [max ?dim t] computes the maximum value of all elements in ndarray [t]. *)

val min : ndarray -> float
(** [min ?dim t] computes the minimum value of all elements in ndarray [t]. *)

val argmax : ndarray -> int
(** [argmax ?dim t] returns a int with the indices of the maximum values of ndarray [t]. *)

val argmin : ndarray -> int
(** [argmin ?dim t] returns a int with the indices of the minimum values of ndarray [t]. *)


val dsum : ndarray -> int -> ndarray
(** [sum dim t] computes the sum of all elements in ndarray [t]. 
    If [dim] is provided, it computes the sum along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val dmean : ndarray -> int -> ndarray
(** [mean dim t] computes the mean of all elements in ndarray [t]. 
    If [dim] is provided, it computes the mean along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val dvar : ndarray -> int -> ndarray
(** [var t] calculates the variance of all elements in the ndarray [t]. 
    If [dim] is provided, it computes the variance along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val dstd : ndarray -> int -> ndarray
(** [std t] calculates the standard deviation of all elements in the ndarray [t]. 
    If [dim] is provided, it computes the std along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val dmax : ndarray -> int -> ndarray
(** [max dim t] computes the maximum value of all elements in ndarray [t]. 
    If [dim] is provided, it computes the maximum along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val dmin : ndarray -> int -> ndarray
(** [min dim t] computes the minimum value of all elements in ndarray [t]. 
    If [dim] is provided, it computes the minimum along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

(* val dargmax : ndarray -> int -> ndarray
(** [argmax dim t] returns a ndarray with the indices of the maximum values along the specified [dim] of ndarray [t]. *)

val dargmin : ndarray -> int -> ndarray
(** [argmin dim t] returns a ndarray with the indices of the minimum values along the specified [dim] of ndarray [t]. *)



val exp : ndarray -> ndarray
(** [exp t] computes the exponential of each element in [t]. *)

val log : ndarray -> ndarray
(** [log t] computes the natural logarithm of each element in [t]. *)

val sqrt : ndarray -> ndarray
(** [sqrt t] computes the square root of each element in [t]. *)

val pow : ndarray -> float -> ndarray
(** [pow t x] raises each element in [t] to the power of [x]. *)

val expand_dims : ndarray -> int -> ndarray
(** [expand_dims t dim] adds a new dimension of size 1 at the specified [dim] in [t]. *)

val squeeze : ndarray -> ndarray
* [squeeze t] removes dimensions of size 1 from [t]. *)
