type t = {
    data: float array
    shape: int list
}

val create : float array -> int list -> t

val zeros : int list -> t

val ones : int list -> t

val rand : int list -> t

val shape : t -> int list

val ndim : t -> int

val at : t -> int list -> float

val add : t -> t -> t

val add_scalar : t -> float -> t

val sub : t -> t -> t

val sub_scalar : t -> float -> t

(** Element-wise multiplication of two ts. *)
val mul : t -> t -> t

val mul_scalar : t -> float -> t

(** Element-wise division of two ts. *)
val div : t -> t -> t

val div_scalar : t -> float -> t

(** Matrix multiplication of two ts. *)
val matmul : t -> t -> t

(** Transpose a t. *)
val transpose : t -> t

(** Reshape a t to the given shape. *)
val reshape : t -> ~shape: int list -> t

(** Convert a t to a float array (only for 1D ts). *)
val to_array : t -> float array
