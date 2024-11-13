type ndarray = {
    data: float array;
    shape: int array;
}
(** An ndarray type [t] containing [data] in a flattened array, and [shape] as a list of dimensions. *)

val create : float array -> int array -> ndarray
(** [create data shape] creates an ndarray [t] with the given [data] and [shape]. *)

val zeros : int array -> ndarray
(** [zeros shape] creates an ndarray of the specified [shape] filled with zeros. *)

val ones : int array -> ndarray
(** [ones shape] creates an ndarray of the specified [shape] filled with ones. *)

val rand : int array -> ndarray
(** [rand shape] creates an ndarray of the specified [shape] with randomly generated values. *)

val shape : ndarray -> int array
(** [shape t] returns the shape of the ndarray [t] as a list of dimensions. *)

val dim : ndarray -> int
(** [dim t] returns the number of dimensions of the ndarray [t]. *)

val get : ndarray -> int array -> float
(** [get t index_list] returns the value of the ndarray [t] at the specified index list[index_list]. *)

val set : ndarray -> int array -> float -> unit
(** [set t idx value] updates the element at index [idx] in [t] with [value]. *)

val add : ndarray -> ndarray -> ndarray
(** [add t1 t2] performs element-wise addition of ndarrays [t1] and [t2]. *)

val add_scalar : ndarray -> float -> ndarray
(** [add_scalar t x] adds scalar [x] to each element in the ndarray [t]. *)

val sub : ndarray -> ndarray -> ndarray
(** [sub t1 t2] performs element-wise subtraction of ndarray [t2] from [t1]. *)

val sub_scalar : ndarray -> float -> ndarray
(** [sub_scalar t x] subtracts scalar [x] from each element in the ndarray [t]. *)

val mul : ndarray -> ndarray -> ndarray
(** [mul t1 t2] performs element-wise multiplication of ndarrays [t1] and [t2]. *)

val mul_scalar : ndarray -> float -> ndarray
(** [mul_scalar t x] multiplies each element in the ndarray [t] by scalar [x]. *)

val div : ndarray -> ndarray -> ndarray
(** [div t1 t2] performs element-wise division of ndarray [t1] by ndarray [t2]. *)

val div_scalar : ndarray -> float -> ndarray
(** [div_scalar t x] divides each element in the ndarray [t] by scalar [x]. *)

val matmul : ndarray -> ndarray -> ndarray
(** [matmul t1 t2] performs matrix multiplication of ndarrays [t1] and [t2]. *)

val transpose : ndarray -> ndarray
(** [transpose t] returns the transpose of ndarray [t]. *)

val reshape : ndarray -> shape:int array -> ndarray
(** [reshape t ~shape] reshapes ndarray [t] to the specified [shape]. *)

val to_array : ndarray -> float array
(** [to_array t] converts the ndarray [t] to a float array. *)

val slice : ndarray -> (int * int) list -> ndarray
(** [slice t ranges] extracts a sub-array from [t] based on [ranges], where each pair in [ranges] specifies the start and end index for each dimension. *)

val fill : ndarray -> float -> unit
(** [fill t value] sets all elements in [t] to the specified [value]. *)

val sum : ndarray -> ?dim:int -> ndarray
(** [sum ?dim t] computes the sum of all elements in ndarray [t]. 
    If [dim] is provided, it computes the sum along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val mean : ndarray -> ?dim:int -> ndarray
(** [mean ?dim t] computes the mean of all elements in ndarray [t]. 
    If [dim] is provided, it computes the mean along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val variance : ndarray -> ?dim:int -> ndarray
(** [variance t] calculates the variance of all elements in the ndarray [t]. 
    If [dim] is provided, it computes the variance along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val std : ndarray -> ?dim:int -> ndarray
(** [std t] calculates the standard deviation of all elements in the ndarray [t]. 
    If [dim] is provided, it computes the std along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val normalize : ndarray -> ?dim:int -> ndarray
(** [normalize t] normalizes tensor [t] by scaling its values to have a mean of 0 and a standard deviation of 1.
    If [dim] is provided, it normalizes along the specified dim, 
    returning a ndarray with those dimensions reduced. *)    

val max : ndarray -> ?dim:int -> ndarray
(** [max ?dim t] computes the maximum value of all elements in ndarray [t]. 
    If [dim] is provided, it computes the maximum along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val min : ndarray -> ?dim:int -> ndarray
(** [min ?dim t] computes the minimum value of all elements in ndarray [t]. 
    If [dim] is provided, it computes the minimum along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val argmax : ndarray -> ?dim:int -> ndarray
(** [argmax ?dim t] returns a ndarray with the indices of the maximum values along the specified [dim] of ndarray [t].
    If [dim] is not provided, it returns the index of the maximum value in the flattened ndarray.
*)

val argmin : ndarray -> ?dim:int -> ndarray
(** [argmin ?dim t] returns a ndarray with the indices of the minimum values along the specified [dim] of ndarray [t].
    If [dim] is not provided, it returns the index of the minimum value in the flattened ndarray.
*)

val exp : ndarray -> ndarray
(** [exp t] computes the exponential of each element in [t]. *)

val log : ndarray -> ndarray
(** [log t] computes the natural logarithm of each element in [t]. *)

val sqrt : ndarray -> ndarray
(** [sqrt t] computes the square root of each element in [t]. *)

val pow : ndarray -> int -> ndarray
(** [pow t x] raises each element in [t] to the power of [x]. *)

val broadcast_to : ndarray -> int array -> ndarray
(** [broadcast_to t shape] broadcasts the ndarray [t] to a new [shape]. This allows operations with different-shaped arrays by expanding [t] as needed. *)

val expand_dims : ndarray -> int -> ndarray
(** [expand_dims t dim] adds a new dimension of size 1 at the specified [dim] in [t]. *)

val squeeze : ndarray -> ndarray
(** [squeeze t] removes dimensions of size 1 from [t]. *)
