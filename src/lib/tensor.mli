(* Type Definitions *)
type ndarray = Ndarray.t

type t = {
  mutable data: ndarray;                    (* The tensor's data as an ndarray *)
  mutable grad: ndarray;             (* Gradient of the tensor, if required *)
  requires_grad: bool;              (* Indicates if the tensor requires gradient computation *)
  mutable backward_fn: (unit -> unit) option;  (* Function to compute gradients for backpropagation *)
  prev: t list;                     (* List of previous tensors for backpropagation *)
}

(* Creation functions *)
val create : data:ndarray -> requires_grad:bool -> prev:t list -> t
(** [create ~data ~requires_grad ~prev] creates a tensor with the specified [data], [requires_grad] flag, and [prev] tensors. *)

val from_ndarray : ?requires_grad:bool -> ndarray -> t
(** [from_ndarray ?requires_grad data] creates a tensor from the given [data] array, with an optional [requires_grad] flag. *)

val scaler : float -> t
(** [scaler x] creates a tensor with a single scalar value [x]. *)

val zeros : int array -> t
(** [zeros shape] creates a tensor filled with zeros, of specified [shape]. *)

val ones : int array -> t
(** [ones shape] creates a tensor filled with ones, of specified [shape]. *)

val arange : float -> t 

val rand : int array -> t
(** [rand shape] creates a tensor with random values in the range [0,1] and the specified [shape]. *)

val xavier_init : int array -> t
(** [xavier_init shape] creates a tensor with Xavier initialization for the given [shape]. *)

val he_init : int array -> t
(** [he_init shape] creates a tensor with He initialization for the given [shape]. *)

(* Tensor information and properties *)
val ndim : t -> int
(** [ndim tensor] returns the number of dimensions of [tensor]. *)

val get_data : t -> ndarray
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
val zero_grad : t -> unit
(** [zero_grad t] resets the gradient of tensor [t] to zero. *)

val accumulate_grad : t -> ndarray -> unit
(** [accumulate_grad t grad] adds the given gradient tensor [grad] to the existing gradient of [t]. *)

val get_grad : t ->  ndarray
(** [get_grad t] retrieves the gradient of tensor [t] as an option type, if it exists. *)

val set_grad : t ->  ndarray -> unit
(** [set_grad t grad] directly sets the gradient of tensor [t] to the given gradient array [grad]. *)

val clip_grad : t -> float -> unit
(** [clip_grad t max_val] clips the gradients of tensor [t] to the range [-max_val, max_val]. *)

(* Element-wise operations *)
val add : t -> t -> t
(** [add t1 t2] performs element-wise addition between tensors [t1] and [t2], supporting broadcasting. *)

val sub : t -> t -> t
(** [sub t1 t2] performs element-wise subtraction between tensors [t1] and [t2]. *)

val mul : t -> t -> t
(** [mul t1 t2] performs element-wise multiplication between tensors [t1] and [t2]. *)

val div : t -> t -> t
(** [div t1 t2] performs element-wise division of tensor [t1] by tensor [t2]. *)

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
val sum : ?dim:int -> t -> t
(** [sum t] calculates the sum of all elements in tensor [t]. *)

val mean : ?dim:int -> t -> t
(** [mean t] calculates the mean of all elements in tensor [t]. *)

(* val variance : t -> t
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

val dmean : t -> int -> t
(** [dmean t dim] calculates the mean of elements along the specified dimension [dim] in tensor [t]. *)

val dvariance : t -> int -> t
(** [dvariance t dim] calculates the variance of elements along the specified dimension [dim] in tensor [t]. *)

val dstd : t -> int -> t
(** [dstd t dim] calculates the standard deviation of elements along the specified dimension [dim] in tensor [t]. *)

val dnormalize : t -> int -> t
* [dnormalize t dim] normalizes tensor [t] along dimension [dim] to have mean 0 and std 1. *)

val dsum : t -> int -> t
(** [dsum t dim] calculates the sum of elements along the specified dimension [dim] in tensor [t]. *)

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

(* need to add more *)
val relu : t -> t
(** [relu t] applies the Rectified Linear Unit function element-wise to tensor [t]. *)

val slice : t -> (int * int) list -> t
(** [slice t ranges] extracts a slice from tensor [t] based on the given ranges for each dimension.
    The [ranges] argument is a list of tuples specifying the start and end indices for slicing along each dimension. *)

val to_string : t -> string
(** [to_string t] converts tensor [t] into its string representation, formatting the data according to its shape and values.
    The result is a human-readable string of the tensor's contents. *)

val softmax : t -> t
(** [softmax t] applies the softmax function to tensor [t], transforming the elements along the last dimension into probabilities.
    Each element is exponentiated and normalized so that the sum of all elements along the dimension equals 1. *)

val log_softmax : t -> t
(** [log_softmax t] applies the logarithm of the softmax function to tensor [t].
    This function is useful for numerically stable computation of log-probabilities, often used in classification problems. *)

val neg : t -> t
(** [neg t] negates each element of tensor [t] element-wise.
    Each element of the tensor is multiplied by -1. *)

val conv2d : t -> t -> stride: int -> padding: int -> t
(** [conv2d input kernel stride padding] performs a 2D convolution operation on the input tensor [input] using the given [kernel].
    The [stride] and [padding] arguments control the sliding stride and padding of the convolution operation. *)

val meanpool2d :t -> kernel_size:int -> stride:int -> t
(** [meanpool2d t ~kernel_size ~stride] performs mean pooling on the input [tensor].
    - [t]: The input tensor of shape [batch_size; channels; height; width].
    - [kernel_size]: The size of the pooling window (e.g., 2 for a 2x2 window).
    - [stride]: The step size for moving the pooling window.
    - Returns a new tensor after mean pooling. *)