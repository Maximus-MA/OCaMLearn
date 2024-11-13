type t = {
    data: float array;
    shape: int list;
}
(** An ndarray type [t] containing [data] in a flattened array, and [shape] as a list of dimensions. *)

val create : float array -> int list -> t
(** [create data shape] creates an ndarray [t] with the given [data] and [shape]. *)

val zeros : int list -> t
(** [zeros shape] creates an ndarray of the specified [shape] filled with zeros. *)

val ones : int list -> t
(** [ones shape] creates an ndarray of the specified [shape] filled with ones. *)

val rand : int list -> t
(** [rand shape] creates an ndarray of the specified [shape] with randomly generated values. *)

val shape : t -> int list
(** [shape t] returns the shape of the ndarray [t] as a list of dimensions. *)

val dim : t -> int
(** [dim t] returns the number of dimensions of the ndarray [t]. *)

val get : t -> int list -> float
(** [get t index_list] returns the value of the ndarray [t] at the specified index list[index_list]. *)

val set : t -> int list -> float -> unit
(** [set t idx value] updates the element at index [idx] in [t] with [value]. *)

val add : t -> t -> t
(** [add t1 t2] performs element-wise addition of ndarrays [t1] and [t2]. *)

val add_scalar : t -> float -> t
(** [add_scalar t x] adds scalar [x] to each element in the ndarray [t]. *)

val sub : t -> t -> t
(** [sub t1 t2] performs element-wise subtraction of ndarray [t2] from [t1]. *)

val sub_scalar : t -> float -> t
(** [sub_scalar t x] subtracts scalar [x] from each element in the ndarray [t]. *)

val mul : t -> t -> t
(** [mul t1 t2] performs element-wise multiplication of ndarrays [t1] and [t2]. *)

val mul_scalar : t -> float -> t
(** [mul_scalar t x] multiplies each element in the ndarray [t] by scalar [x]. *)

val div : t -> t -> t
(** [div t1 t2] performs element-wise division of ndarray [t1] by ndarray [t2]. *)

val div_scalar : t -> float -> t
(** [div_scalar t x] divides each element in the ndarray [t] by scalar [x]. *)

val matmul : t -> t -> t
(** [matmul t1 t2] performs matrix multiplication of ndarrays [t1] and [t2]. *)

val transpose : t -> t
(** [transpose t] returns the transpose of ndarray [t]. *)

val reshape : t -> shape:int list -> t
(** [reshape t ~shape] reshapes ndarray [t] to the specified [shape]. *)

val to_array : t -> float array
(** [to_array t] converts the ndarray [t] to a float array. *)

val slice : t -> (int * int) list -> t
(** [slice t ranges] extracts a sub-array from [t] based on [ranges], where each pair in [ranges] specifies the start and end index for each dimension. *)

val fill : t -> float -> unit
(** [fill t value] sets all elements in [t] to the specified [value]. *)

val sum : t -> ?dim:int -> t
(** [sum ?dim t] computes the sum of all elements in ndarray [t]. 
    If [dim] is provided, it computes the sum along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val mean : t -> ?dim:int -> t
(** [mean ?dim t] computes the mean of all elements in ndarray [t]. 
    If [dim] is provided, it computes the mean along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val variance : t -> ?dim:int -> t
(** [variance t] calculates the variance of all elements in the ndarray [t]. 
    If [dim] is provided, it computes the variance along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val std : t -> ?dim:int -> t
(** [std t] calculates the standard deviation of all elements in the ndarray [t]. 
    If [dim] is provided, it computes the std along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val normalize : t -> ?dim:int -> t
(** [normalize t] normalizes tensor [t] by scaling its values to have a mean of 0 and a standard deviation of 1.
    If [dim] is provided, it normalizes along the specified dim, 
    returning a ndarray with those dimensions reduced. *)    

val max : t -> ?dim:int -> t
(** [max ?dim t] computes the maximum value of all elements in ndarray [t]. 
    If [dim] is provided, it computes the maximum along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val min : t -> ?dim:int -> t
(** [min ?dim t] computes the minimum value of all elements in ndarray [t]. 
    If [dim] is provided, it computes the minimum along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val argmax : t -> ?dim:int -> t
(** [argmax ?dim t] returns a ndarray with the indices of the maximum values along the specified [dim] of ndarray [t].
    If [dim] is not provided, it returns the index of the maximum value in the flattened ndarray.
*)

val argmin : t -> ?dim:int -> t
(** [argmin ?dim t] returns a ndarray with the indices of the minimum values along the specified [dim] of ndarray [t].
    If [dim] is not provided, it returns the index of the minimum value in the flattened ndarray.
*)

val exp : t -> t
(** [exp t] computes the exponential of each element in [t]. *)

val log : t -> t
(** [log t] computes the natural logarithm of each element in [t]. *)

val sqrt : t -> t
(** [sqrt t] computes the square root of each element in [t]. *)

val pow : t -> int -> t
(** [pow t x] raises each element in [t] to the power of [x]. *)

val broadcast_to : t -> int list -> t
(** [broadcast_to t shape] broadcasts the ndarray [t] to a new [shape]. This allows operations with different-shaped arrays by expanding [t] as needed. *)

val expand_dims : t -> int -> t
(** [expand_dims t dim] adds a new dimension of size 1 at the specified [dim] in [t]. *)

val squeeze : t -> t
(** [squeeze t] removes dimensions of size 1 from [t]. *)
