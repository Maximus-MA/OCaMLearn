(** The type representing an optimizer. *)
type tensor = Tensor.t

type t = {
  parameters: tensor list;  (* List of tensors representing the parameters to optimize. *)
  step: unit -> unit;       (* Function to update parameters based on their gradients. *)
  zero_grad: unit -> unit;  (* Function to reset all gradients to zero. *)
}

val create_SGD : params:tensor list -> lr:float -> t
(** [create_SGD params lr] creates an SGD (Stochastic Gradient Descent) optimizer for the list of tensors [params] with the specified learning rate [lr]. *)

val create_Adam : params:tensor list -> lr:float -> beta1:float -> beta2:float -> eps:float -> t
(** [create_Adam params lr beta1 beta2 eps] creates an Adam optimizer for the list of tensors [params] with the specified learning rate [lr], first moment decay rate [beta1],
    second moment decay rate [beta2], and small constant [eps] to prevent division by zero. *)

val step : t -> unit
(** [step optimizer] performs a single optimization step and updates the parameters of the optimizer in place. *)

val zero_grad : t -> unit
(** [zero_grad optimizer] resets all gradients of the parameters to zero, clearing the gradients of all parameters. *)
