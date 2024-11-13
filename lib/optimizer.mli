(** The type representing an optimizer. *)
type t

(** Create an SGD optimizer. *)
val sgd : params:tensor list -> lr:float -> t

(** Create an Adam optimizer. *)
val adam :
  params:tensor list ->
  lr:float ->
  beta1:float ->
  beta2:float ->
  eps:float ->
  t

(** Perform an optimization step. *)
val step : t -> unit

(** Zero the gradients of all parameters. *)
val zero_grad : t -> unit
