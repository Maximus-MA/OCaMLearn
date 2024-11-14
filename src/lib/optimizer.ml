(* src/optimizer.ml *)
[@@@ocaml.warning "-27"]

type tensor = Tensor.t

type t = {
  parameters: tensor list;         (* List of tensors representing the optimizer's parameters *)
  step: unit -> unit;              (* Function to update parameters *)
  zero_grad: unit -> unit;         (* Function to reset gradients *)
}

(* Placeholder for unimplemented functions *)
let not_implemented feature_name =
  failwith (feature_name ^ " is not yet implemented")

let create_SGD ~params ~lr =
  not_implemented "create_SGD"

let create_Adam ~params ~lr ~beta1 ~beta2 ~eps =
  not_implemented "create_Adam"

let step optimizer =
  not_implemented "step"

let zero_grad optimizer =
  not_implemented "zero_grad"