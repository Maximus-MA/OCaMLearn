(* src/optimizer.ml *)
[@@@ocaml.warning "-27"]
open Core
type tensor = Tensor.t

type t = {
  parameters: tensor list;         (* List of tensors representing the optimizer's parameters *)
  step: unit ->unit;              (* Function to update parameters *)
  zero_grad: unit -> unit;         (* Function to reset gradients *)
}

(* Placeholder for unimplemented functions *)
let not_implemented feature_name =
  failwith (feature_name ^ " is not yet implemented")

let create_SGD ~params ~lr =
  let step = fun () -> List.iter params ~f:(fun param -> Tensor.(param.data <- 
    Ndarray.sub param.data (Ndarray.mul param.grad (Ndarray.scaler lr)))) in
  let zero_grad = fun () -> List.iter params ~f:(fun param -> Tensor.zero_grad param) in
    { parameters = params; step = step; zero_grad = zero_grad }

let create_Adam ~params ~lr ~beta1 ~beta2 ~eps =
  not_implemented "create_Adam"