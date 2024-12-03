open Core

let () =
  let input = Tensor.rand [|2; 100|] in
  let model = Model.create_Sequential [
    Model.create_Linear ~in_features:100 ~out_features:50 ~bias:true;
    Model.create_ReLU ();
    Model.create_Linear ~in_features:50 ~out_features:10 ~bias:true;
  ] in 
  let output = Model.forward model [input] in
  Printf.printf "Output: %s\n" (Tensor.to_string output);
  let loss_func = Model.create_CrossEntropy () in
  let target = Tensor.zeros [|2; 10|] in
  let loss = Model.forward loss_func [output; target] in
  Printf.printf "Loss: %s\n" (Tensor.to_string loss);
  Utils.backprop loss;
  Printf.printf "Gradients: %s\n" Ndarray.(to_string input.grad);

