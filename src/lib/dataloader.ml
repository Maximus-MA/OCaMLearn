(* src/dataloader.ml *)

[@@@ocaml.warning "-27"]


type dataset = Dataset.t

type tensor = Tensor.t

type transform = Transform.t

type tensor_dataset = {
  data : tensor;
  label : tensor;
}

type t = {
  dataset : tensor_dataset;  (* The dataset to load from *)
  batch_size : int;          (* Number of samples per batch *)
  total_batches : int;       (* Total number of batches in the dataset *)
}

(* Placeholder for unimplemented functions *)
let not_implemented feature_name =
  failwith (feature_name ^ " is not yet implemented")

let create dataset ~batch_size ~shuffle ?transorms =
  not_implemented "create"

let get_batch loader idx =
  not_implemented "get_batch"

let get_total_batches loader =
  not_implemented "get_total_batches"