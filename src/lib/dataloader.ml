(* src/dataloader.ml *)

[@@@ocaml.warning "-27"]

type ndarray = Ndarray.t

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
(* let not_implemented feature_name =
  failwith (feature_name ^ " is not yet implemented") *)

let create ?(transforms = []) dataset ~batch_size ~shuffle  =
  let apply_transforms (ndarray: ndarray) (transforms: transform list) =
    List.fold_left (fun acc transform -> transform acc) ndarray transforms
  in
  if shuffle then 
    let shuffled_dataset = Dataset.shuffle dataset in
    let tensor_dataset = 
      { data = Tensor.from_ndarray (apply_transforms shuffled_dataset.data transforms); 
      label = Tensor.from_ndarray shuffled_dataset.label }
    in let num_samples = shuffled_dataset.data.shape.(0) in 
    { dataset = tensor_dataset; batch_size = batch_size; total_batches = (num_samples + batch_size - 1) / batch_size }
  else
    let tensor_dataset = 
    { data = Tensor.from_ndarray (apply_transforms dataset.data transforms); 
    label = Tensor.from_ndarray dataset.label }
    in let num_samples = dataset.data.shape.(0) in  
    { dataset = tensor_dataset; batch_size = batch_size; total_batches = (num_samples + batch_size - 1) / batch_size }

let get_batch loader idx =
  {data = Tensor.slice (loader.dataset.data) [idx*loader.batch_size, idx*loader.batch_size+loader.batch_size];
  label = Tensor.slice (loader.dataset.label) [idx*loader.batch_size, idx*loader.batch_size+loader.batch_size]}

let get_total_batches loader =
  loader.total_batches