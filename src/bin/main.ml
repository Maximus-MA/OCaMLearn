open Core

let example1 () =
  let input = Tensor.ones [|2; 100|] in
  let model = Model.create_Sequential [
    Model.create_Linear ~in_features:100 ~out_features:50 ~bias:true;
    Model.create_ReLU ();
    Model.create_Linear ~in_features:50 ~out_features:10 ~bias:true;
  ] in 
  let output = Model.forward model [input] in
  Printf.printf "Input: %s\n" (Tensor.to_string input);
  Printf.printf "Output: %s\n" (Tensor.to_string output);
  let loss_func = Model.create_CrossEntropy () in
  let target = Tensor.rand [|2; 10|] in
  let loss = Model.forward loss_func [output; target] in
  Printf.printf "Loss: %s\n" (Tensor.to_string loss);
  Utils.backprop loss;
  Printf.printf "Gradients: %s\n" Ndarray.(to_string input.grad)

let example2 () =
  let input = Tensor.ones [|2; 100|] in
  (* Printf.printf "Input: %s\n" (Tensor.to_string input); *)
  let model = Model.create_Sequential [
    Model.create_Linear ~in_features:100 ~out_features:50 ~bias:true;
    Model.create_ReLU ();
    Model.create_Linear ~in_features:50 ~out_features:10 ~bias:true;
    ] in 
  let loss_func = Model.create_CrossEntropy () in
  let target =  Tensor.zeros [|2; 10|] in
  Tensor.set target [|0; 0|] 1.;
  Tensor.set target [|1; 2|] 1.;
  let optimizer = Optimizer.create_SGD ~params:Model.(model.parameters) ~lr:0.01 in
  (* Printf.printf "Target: %s\n" (Tensor.to_string target); *)
  for _ = 0 to 10 do
    let output = Model.forward model [input] in
    (* Printf.printf "Output: %s\n" (Tensor.to_string output); *)
    let loss = Model.forward loss_func [output; target] in
    Printf.printf "Loss: %s\n" (Tensor.to_string loss);
    optimizer.zero_grad ();
    Utils.backprop loss;
    (* Printf.printf "Gradients: %s\n" Ndarray.(to_string input.grad); *)
    optimizer.step ()
  done


let () =
  example1 ();
  example2 ()
  
