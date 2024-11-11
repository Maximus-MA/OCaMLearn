(** The type representing an optimizer. *)
type optimizer

(** Create an SGD optimizer. *)
val sgd : params:tensor list -> lr:float -> optimizer

(** Create an Adam optimizer. *)
val adam :
  params:tensor list ->
  lr:float ->
  beta1:float ->
  beta2:float ->
  eps:float ->
  optimizer

(** Perform an optimization step. *)
val step : optimizer -> unit

(** Zero the gradients of all parameters. *)
val zero_grad : optimizer -> unit
