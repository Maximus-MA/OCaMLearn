(* src/transform.ml *)

[@@@ocaml.warning "-27"]

type ndarray = Ndarray.t
(* Represents a transformation applied to a ndarray. *)
type t = ndarray -> ndarray

(* Placeholder for unimplemented functions *)
let not_implemented feature_name =
  failwith (feature_name ^ " is not yet implemented")

let normalize x =
  Ndarray.normalize x

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

let image_scale t =
  Ndarray.image_scale t