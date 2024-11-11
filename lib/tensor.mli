type ndarray = Ndarray.t
type t = {
    data: ndarray;
    grad: ndarray;
    requires_grad: bool;
    backward_fn: (unit -> unit) option;
    prev: t list;
}

(** {1 Tensor Creation} *)

(** Create a tensor from a given ndarray. *)
val from_ndarray : ?requires_grad:bool -> ndarray -> t

(** Create a tensor filled with zeros of the specified shape. *)
val zeros : ?requires_grad:bool -> int list -> t

(** Create a tensor filled with ones of the specified shape. *)
val ones : ?requires_grad:bool -> int list -> t

val rand : ?requires_grad:bool -> int list -> t

(** Create a tensor from a float array. *)

(** {1 Tensor Properties} *)

val ndim : t -> int

val at : t -> int list -> float

(** Get the data of the tensor as an ndarray. *)
val to_ndarray : t -> ndarray

(** Get the gradient of the tensor as an ndarray. *)
val get_grad : t -> ndarray

(** Check if the tensor requires gradient computation. *)
val get_requires_grad : t -> bool

val set_requires_grad : t -> bool -> t

(** Get the shape of the tensor. *)

(** Reset the gradient of the tensor to zero. *)
val zero_grad : t -> unit

(** {1 Tensor Operations} *)

(** Add two tensors element-wise. Supports broadcasting. *)
val add : t -> t -> t

(** Add a scalar to all elements of the tensor. *)
val add_scalar : t -> float -> t

(** Subtract two tensors element-wise. Supports broadcasting. *)
val sub : t -> t -> t

(** Subtract a scalar from all elements of the tensor. *)
val sub_scalar : t -> float -> t

(** Multiply two tensors element-wise. Supports broadcasting. *)
val mul : t -> t -> t

(** Multiply all elements of the tensor by a scalar. *)
val mul_scalar : t -> float -> t

(** Divide two tensors element-wise. Supports broadcasting. *)
val div : t -> t -> t

(** Divide all elements of the tensor by a scalar. *)
val div_scalar : t -> float -> t

(** Perform matrix multiplication between two tensors. *)
val matmul : t -> t -> t

(** Transpose the tensor. *)
val transpose : t -> t

(** Reshape the tensor to the specified shape. *)
val reshape : t -> shape:int list -> t
