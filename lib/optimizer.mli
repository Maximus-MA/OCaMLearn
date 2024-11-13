type t = {
	step: step : t -> unit;
	zero_grad: zero_grad : t -> unit;
}

(* * Create an SGD optimizer. *)
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


(** Zero the gradients of all parameters. *)
