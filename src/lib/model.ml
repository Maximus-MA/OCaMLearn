[@@@ocaml.warning "-27"]

open Core

type tensor = Tensor.t

type t = {
  parameters: tensor list;                (* List of tensors representing the layer's parameters *)
  forward_fn: tensor list -> tensor; (* Forward function for layer computation *)
} [@@deriving show]

(* Placeholder for unimplemented functions *)
(* let not_implemented feature_name =
  failwith (feature_name ^ " is not yet implemented") *)

let create ~parameters ~forward_fn =
  { parameters; forward_fn }

let forward layer = layer.forward_fn
  
let get_parameters layer =
  layer.parameters

(* ignore bias option *)
let create_Linear ~in_features ~out_features ~bias =
  let w = Tensor.rand [|in_features; out_features|] in
  if bias then
    let b = Tensor.rand [|1; out_features|] in
    let parameters = [w; b] in
    let forward_fn inputs =
      let x = List.hd_exn inputs in
      Tensor.(add (matmul x w) b) in
    create ~parameters ~forward_fn
  else
    let parameters = [w] in
    let forward_fn inputs =
      let x = List.hd_exn inputs in
      Tensor.matmul x w in
    create ~parameters ~forward_fn

let create_Conv2d ~in_channels ~out_channels ~kernel_size ~stride ~padding ~bias =
  Printf.printf "create conv2d\n";
  let w = Tensor.rand [|out_channels; in_channels; kernel_size; kernel_size|] in
  Printf.printf "Kernel shape %d\n" (Array.length w.data.shape);
  if bias then
    let b = Tensor.rand [|1; out_channels|] in
    let parameters = [w; b] in
    let forward_fn inputs =
      let x = List.hd_exn inputs in
      Tensor.(add (Tensor.conv2d x w ~stride ~padding) b) in
    create ~parameters ~forward_fn
  else
    let parameters = [w] in
    let forward_fn inputs =
      let x = List.hd_exn inputs in
      Tensor.conv2d x w ~stride ~padding in
    create ~parameters ~forward_fn

let create_MeanPool2d ~kernel_size ~stride =
  let parameters = [] in
  let forward_fn inputs =
    let x = List.hd_exn inputs in
    Tensor.meanpool2d x ~kernel_size ~stride in
  create ~parameters ~forward_fn

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
  let forward_fn inputs =
    let x = List.hd_exn inputs in
    Tensor.(div (scaler 1.0) (add (scaler 1.0) (exp (neg x)))) in
  create ~parameters:[] ~forward_fn

let create_Softmax () =
  let forward_fn inputs =
    let x = List.hd_exn inputs in
    Tensor.softmax x
  in
  create ~parameters:[] ~forward_fn
  

let create_MSE () =
  let forward_fn inputs =
    let logits = List.nth_exn inputs 0 in
    let targets = List.nth_exn inputs 1 in
    let loss = Tensor.mean (Tensor.pow (Tensor.sub logits targets) 2) in
    loss
  in
  create ~parameters:[] ~forward_fn

let create_CrossEntropy () =
  let forward_fn inputs =
    let logits = List.nth_exn inputs 0 in
    let targets = List.nth_exn inputs 1 in
    let log_probs = Tensor.log_softmax logits in
  (* Printf.printf "hello"; *)
    let loss = Tensor.neg (Tensor.mean ~dim:0 (Tensor.dsum (Tensor.mul targets log_probs) (Ndarray.dim logits.data - 1))) in
    loss
  in
  create ~parameters:[] ~forward_fn
 
