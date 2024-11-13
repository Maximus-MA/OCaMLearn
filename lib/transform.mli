type tensor = Tensor.t

type t = {
    transform : tensor -> tensor;
}

val transform : tensor -> tensor
(** [transform t x] applies the transformation [t] to the input tensor [x]. *)

val create_normalize : unit -> t
(** [create ()] creates a new transformation. *)

val create_totensor : unit -> t
(** [create ()] creates a new transformation. *)