(* src/transform.ml *)

[@@@ocaml.warning "-27"]

type tensor = Tensor.t

type t = tensor -> tensor
(** The type for transformations *)

(* Placeholder for unimplemented functions *)
let not_implemented feature_name =
  failwith (feature_name ^ " is not yet implemented")

let normalize =
  not_implemented "normalize"

let resize width height =
  not_implemented "resize"

let rotate angle =
  not_implemented "rotate"

let translate x y =
  not_implemented "translate"

let scale x y =
  not_implemented "scale"

let flip horizontal vertical =
  not_implemented "flip"