(** The type representing an optimizer. *)
type tensor = Tensor.t

type t = {
  parameters: tensor list;
  step: unit -> unit;
  zero_grad: unit -> unit;
}

val create_SGD : params:tensor list -> lr:float -> t

val create_Adam : params:tensor list -> lr:float -> beta1:float -> beta2:float -> eps:float -> t

val step : t -> unit

val zero_grad : t -> unit