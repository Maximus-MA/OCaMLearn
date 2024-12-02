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
  let open Random in
  let data = dataset.data.data in
  let label = dataset.label.data in
  let data_shape = dataset.data.shape in
  let label_shape = dataset.label.shape in
  if data_shape.(0) <> label_shape.(0) then
    invalid_arg "Data and label must have the same number of samples!";
  let num_samples = data_shape.(0) in
  let sample_size_data = Array.fold_left ( * ) 1 (Array.sub data_shape 1 (Array.length data_shape - 1)) in
  let sample_size_label =
    if Array.length label_shape = 1 then 1
    else Array.fold_left ( * ) 1 (Array.sub label_shape 1 (Array.length label_shape - 1))
  in
  let indices = Array.init num_samples (fun i -> i) in
  for i = num_samples - 1 downto 1 do
    let j = int (i + 1) in
    let temp = indices.(i) in
    indices.(i) <- indices.(j);
    indices.(j) <- temp
  done;
  let shuffled_data = Array.make (Array.length data) 0.0 in
  let shuffled_label = Array.make (Array.length label) 0.0 in
  for i = 0 to num_samples - 1 do
    let src_idx = indices.(i) in
    Array.blit data (src_idx * sample_size_data) shuffled_data (i * sample_size_data) sample_size_data;
    Array.blit label (src_idx * sample_size_label) shuffled_label (i * sample_size_label) sample_size_label
  done;
  {
    data = { data = shuffled_data; shape = data_shape };
    label = { data = shuffled_label; shape = label_shape };
  }

let split dataset ratio =
  not_implemented "split"