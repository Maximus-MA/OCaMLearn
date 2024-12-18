[@@@ocaml.warning "-27"]

type ndarray = Ndarray.t
(* Represents a transformation applied to a ndarray. *)
type t = ndarray -> ndarray

let normalize x =
  Ndarray.normalize x

let image_scale t =
  Ndarray.image_scale t