[@@@ocaml.warning "-27"]

open Core

type tensor = Tensor.t

type t = {
  parameters: tensor list;                (* List of tensors representing the layer's parameters *)
  forward_fn: tensor list -> tensor; (* Forward function for layer computation *)
} [@@deriving show]

(* Placeholder for unimplemented functions *)
let not_implemented feature_name =
  failwith (feature_name ^ " is not yet implemented")

let create ~parameters ~forward_fn =
  { parameters; forward_fn }

let forward layer = layer.forward_fn
  
let get_parameters layer =
  layer.parameters

(* ignore bias option *)
let create_Linear ~in_features ~out_features ~bias =
  let w = Tensor.rand [|out_features; in_features|] in
  let b = Tensor.rand [|out_features|] in
  let parameters = [w; b] in
  let forward_fn inputs =
    let x = List.hd_exn inputs in
    Tensor.(add (matmul w x) b) in
  create ~parameters ~forward_fn

let create_Conv2d ~in_channels ~out_channels ~kernel_size ~stride ~padding =
  not_implemented "create_Conv2d"

let create_Flatten () =
  let parameters = [] in
  let forward_fn inputs =
    let x = List.hd_exn inputs in
    let batch = Array.get (Tensor.shape x) 0 in
    Tensor.reshape x ~shape:[|batch; -1|] in
  create ~parameters ~forward_fn


let create_Sequential layers =
  let parameters = List.concat (List.map ~f:get_parameters layers) in
  let forward_fn inputs = layers |>
    List.fold 
      ~init: (List.hd_exn inputs)
      ~f: (fun acc layer -> forward layer [acc]) in
  create ~parameters ~forward_fn

let create_ReLU () =
  let forward_fn inputs = 
    let x = List.hd_exn inputs in
    Tensor.relu x in
  create ~parameters:[] ~forward_fn


let create_Sigmoid () =
  not_implemented "create_Sigmoid"

let create_LeakyReLU ?alpha =
  not_implemented "create_LeakyReLU"

let create_Tanh () =
  not_implemented "create_Tanh"

let create_Softmax () =
  let forward_fn inputs =
    let x = List.hd_exn inputs in
    Tensor.softmax x
  in
  create ~parameters:[] ~forward_fn
  

let create_MSE () =
  not_implemented "create_MSE"

  let create_CrossEntropy () =
    let forward_fn inputs =
      let logits = List.nth_exn inputs 0 in
      let targets = List.nth_exn inputs 1 in
      let log_probs = Tensor.log_softmax logits in
      let loss = Tensor.neg (Tensor.mean (Tensor.dsum (Tensor.mul targets log_probs) (Ndarray.dim logits.data - 1))) in
      loss
    in
    create ~parameters:[] ~forward_fn
  