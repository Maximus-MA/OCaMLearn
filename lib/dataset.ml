(* src/dataset.ml *)

[@@@ocaml.warning "-27"]

type ndarray = Ndarray.t

type t = {
  data : ndarray;  (* The data for the dataset *)
  label : ndarray; (* The labels for the dataset *)
}

(* Placeholder for unimplemented functions *)
let not_implemented feature_name =
  failwith (feature_name ^ " is not yet implemented")

let get_item dataset idx =
  not_implemented "get_item"

let shuffle dataset =
  not_implemented "shuffle"

let split dataset ratio =
  not_implemented "split"