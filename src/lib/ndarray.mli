type t = {
  data: float array;  (* The underlying data array of the ndarray. *)
  shape: int array;   (* The shape of the ndarray. *)
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
(** [matmul a b] performs matrix multiplication on ndarrays [a] and [b], including batch dimensions. *)

val create : float array -> int array -> t
(** [create data shape] creates a new ndarray with the given [data] and [shape]. *)

val zeros : int array -> t
(** [zeros shape] creates an ndarray of the given [shape] filled with zeros. *)

val ones : int array -> t
(** [ones shape] creates an ndarray of the given [shape] filled with ones. *)

val arange : float -> t 

val rand : int array -> t
(** [rand shape] creates an ndarray of the given [shape] filled with random values between 0 and 1. *)

val randn : int array -> t
(** [randn shape] creates an ndarray of the given [shape] filled with random values drawn from a standard normal distribution. *)

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
(** [set t idx value] updates the element at index [idx] in [t] with [value]. *)

val transpose : t -> t
(** [transpose t] returns the transpose of ndarray [t]. *)

val to_array : t -> float array
(** [to_array t] converts the ndarray [t] to a float array. *)

val slice : t -> (int * int) list -> t
(** [slice t ranges] extracts a sub-array from [t] based on [ranges], where each pair in [ranges] specifies the start and end index for each dimension. *)

val fill : t -> float -> unit
(** [fill t value] sets all elements in [t] to the specified [value]. *)

val sum : t -> float
(** [sum t] computes the sum of all elements in ndarray [t]. *)

val mean : t -> float
(** [mean t] computes the mean of all elements in ndarray [t]. *)

val var : t -> float
(** [var t] calculates the variance of all elements in the ndarray [t]. *)

val std : t -> float
(** [std t] calculates the standard deviation of all elements in the ndarray [t]. *)

val max : t -> float
(** [max t] computes the maximum value of all elements in ndarray [t]. *)

val min : t -> float
(** [min t] computes the minimum value of all elements in ndarray [t]. *)

val argmax : t -> int
(** [argmax t] returns the index of the maximum value in ndarray [t]. *)

val dargmax : t -> int -> t
(** [dargmax dim t] computes the index of the maximum value along the specified dimension [dim] in ndarray [t]. *)

val argmin : t -> int
(** [argmin t] returns the index of the minimum value in ndarray [t]. *)

val dsum : t -> int -> t
(** [dsum dim t] computes the sum of all elements in ndarray [t]. 
    If [dim] is provided, it computes the sum along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val dmean : t -> int -> t
(** [dmean dim t] computes the mean of all elements in ndarray [t]. 
    If [dim] is provided, it computes the mean along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val dvar : t -> int -> t
(** [dvar t] calculates the variance of all elements in the ndarray [t]. 
    If [dim] is provided, it computes the variance along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val dstd : t -> int -> t
(** [dstd t] calculates the standard deviation of all elements in the ndarray [t]. 
    If [dim] is provided, it computes the std along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val dmax : t -> int -> t
(** [dmax dim t] computes the maximum value of all elements in ndarray [t]. 
    If [dim] is provided, it computes the maximum along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val dmin : t -> int -> t
(** [dmin dim t] computes the minimum value of all elements in ndarray [t]. 
    If [dim] is provided, it computes the minimum along the specified dim, 
    returning a ndarray with those dimensions reduced. *)

val exp : t -> t
(** [exp t] computes the exponential of each element in [t]. *)

val log : t -> t
(** [log t] computes the natural logarithm of each element in [t]. *)

val sqrt : t -> t
(** [sqrt t] computes the square root of each element in [t]. *)

val pow : t -> float -> t
(** [pow t x] raises each element in [t] to the power of [x]. *)

val expand_dims : t -> int -> t
(** [expand_dims t dim] adds a new dimension of size 1 at the specified [dim] in [t]. *)

val squeeze : t -> t
(** [squeeze t] removes dimensions of size 1 from [t]. *)

val pad_shape_to : int array -> int array -> int array * int array
(** [pad_shape_to src_shape target_shape] computes the necessary paddings to transform 
    [src_shape] into [target_shape]. Returns a pair of arrays representing 
    pre-padding and post-padding for each dimension. *)

val map : t -> f:(float -> float) -> t
(** [map t ~f] applies the function [f] to each element of the ndarray [t] 
    and returns a new ndarray with the resulting values. *)

val reduce_sum_to_shape : t -> int array -> t
(** [reduce_sum_to_shape t shape] reduces the ndarray [t] to the specified [shape] 
    by summing over the necessary dimensions. *)

val negate : t -> t
(** [negate t] negates each element of the ndarray [t], 
    returning a new ndarray where each element is the negative of the corresponding input element. *)

val relu : t -> t
(** [relu t] applies the ReLU (Rectified Linear Unit) function to each element of the ndarray [t], 
    replacing negative values with 0 and leaving non-negative values unchanged. *)

val to_string : t -> string
(** [to_string t] converts the ndarray [t] to its string representation 
    for easier visualization and debugging. *)

val scaler : float -> t
(** [scaler x] creates a scalar ndarray containing the single value [x]. *)

val normalize : t -> t
(** [normalize t] normalizes the ndarray [t] by subtracting the mean and dividing by the standard deviation for each feature.
    It returns a new ndarray with the normalized values. *)

val conv2d : t -> t -> stride: int -> padding: int -> t
(** [conv2d input kernel stride padding] performs a 2D convolution on the input [input] with the given [kernel], [stride], and [padding].
    It returns the convolved output. *)

val transpose_last_two_dims : t -> t
(** [transpose_last_two_dims t] transposes the last two dimensions of the ndarray [t]. *)

val flip_and_swap_kernel : t -> t
(** [flip_and_swap_kernel kernel] swaps the first two dimensions of the input [kernel] and rotates the last two dimensions by 180 degrees.
    It returns a new ndarray with the shape (in_channels, out_channels, kernel_height, kernel_width). *)

val layerwise_convolution_with_doutput_as_kernel : t -> t -> int -> int -> t
(** [layerwise_convolution_with_doutput_as_kernel input doutput stride padding] performs a layerwise convolution operation
    where [doutput] is used as the kernel and convolves with [input] tensor for each channel separately.
    It returns a new ndarray with shape (batch_size, out_channel, input_channel, kernel_h, kernel_w), where 
    [kernel_h] and [kernel_w] are computed based on the input and output dimensions, and the convolution is performed 
    with the given [stride] and [padding]. *)

val image_scale : t -> t
(** [image_scale t] scales the image to [-1,1] *)

val l2_norm : t -> float
(** [l2_norm t] returns the L2 norm of the input tensor [t]. *)

val clip_by_norm : t -> float -> unit
(** [clip_by_norm t value] clips the values of the input ndarray [t] to be within the range [-value, value]. *)

val clip_by_range : t -> float -> float -> unit
(** [clip_by_range t min max] clips the values of the input ndarray [t] to be within the range [min, max]. *)