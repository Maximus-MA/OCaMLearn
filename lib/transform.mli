type tensor = Tensor.t

type t =  tensor -> tensor
(** The type for transformations. *)

val normalize : t
(** [create ()] creates a new transformation. *)

val resize : int -> int -> t
(** [resize width height] creates a new transformation that resizes the input tensor to the specified [width] and [height]. *)

val rotate : float -> t
(** [rotate angle] creates a new transformation that rotates the input tensor by the specified [angle]. *)

val translate : float -> float -> t
(** [translate x y] creates a new transformation that translates the input tensor by the specified [x] and [y] offsets. *)

val scale : float -> float -> t
(** [scale x y] creates a new transformation that scales the input tensor by the specified [x] and [y] factors. *)

val flip : bool -> bool -> t
(** [flip horizontal vertical] creates a new transformation that flips the input tensor horizontally and/or vertically based on the specified options. *)
