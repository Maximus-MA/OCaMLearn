(* src/backprop.ml *)

[@@@ocaml.warning "-27"]

type tensor = Tensor.t

(* Placeholder for unimplemented functions *)
let not_implemented feature_name =
  failwith (feature_name ^ " is not yet implemented")

let backprop tensor =
  not_implemented "backprop"