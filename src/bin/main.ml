(* open Core *)

(* let example1 () =
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
  done *)

(* let print_shape shape =
  Printf.printf "[%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int shape)))
;;

(* Helper function to print the data *)
let print_data data =
  Printf.printf "[%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_float data)))
;; *)
(* let example3 () =
  let dataset = Dataset.load_csv "dataset/iris.csv" 4 label_string_to_int 3 in
  print_data dataset.data.data;
  print_shape dataset.data.shape;
  let dataset = Dataset.shuffle dataset in
  let train_dataset, test_dataset = Dataset.split dataset 0.8 in
  let train_input = Tensor.from_ndarray train_dataset.data in
  let train_target = Tensor.from_ndarray train_dataset.label in
  let test_input = Tensor.from_ndarray test_dataset.data in
  let test_target = Tensor.from_ndarray test_dataset.label in
  (* let train_loader = Dataloader.create train_dataset ~batch_size:10 ~shuffle:true in
  let test_loader = Dataloader.create test_dataset ~batch_size:10 ~shuffle:false in *)

  let model = Model.create_Sequential [
    Model.create_Linear ~in_features:4 ~out_features:50 ~bias:true;
    Model.create_ReLU ();
    Model.create_Linear ~in_features:50 ~out_features:3 ~bias:true;
  ] in 

  let loss_func = Model.create_CrossEntropy () in
  let optimizer = Optimizer.create_SGD ~params:Model.(model.parameters) ~lr:0.01 in

  let train () =
    for _ = 1 to 30 do
      (* for batch_idx = 0 to Dataloader.get_total_batches train_loader - 1 do *)
        (* let batch = Dataloader.get_batch train_loader batch_idx in *)
        (* Printf.printf "Input: %s\n" (Tensor.to_string batch.data); *)
        (* Printf.printf "Target: %s\n" (Tensor.to_string batch.label); *)
        
        let output = Model.forward model [train_input] in
        (* Printf.printf "Output: %s\n" (Tensor.to_string output); *)
        let loss = Model.forward loss_func [output; train_target] in
        Printf.printf "Loss: %s\n"  (Tensor.to_string loss);
        optimizer.zero_grad ();
        Utils.backprop loss;
        optimizer.step ()
      (* done *)
    done
  in

  let test () =
    (* for batch_idx = 0 to Dataloader.get_total_batches test_loader - 1 do *)
      (* let batch = Dataloader.get_batch test_loader batch_idx in *)
      let output = Model.forward model [test_input] in
      Printf.printf "Output: %s\n" (Tensor.to_string output);

      let loss = Model.forward loss_func [output; test_target] in
      Printf.printf "Target: %s\n" (Tensor.to_string test_target);
      (* Printf.printf "Test Batch: %d, Loss: %s\n" batch_idx (Tensor.to_string loss); *)
      Printf.printf "Test Loss: %s\n" (Tensor.to_string loss);
    (* done *)
  in

  train ();
  test () *)
let label_string_to_int label_string = 
  match label_string with
  | "Iris-setosa" -> 0
  | "Iris-versicolor" -> 1
  | "Iris-virginica" -> 2
  | _ -> failwith "Unknown label"
let example_iris () =
  let dataset = Dataset.load_csv "dataset/iris.csv" 4 label_string_to_int 3 in
  (* print_data dataset.data.data; *)
  (* print_shape dataset.data.shape; *)
  let dataset = Dataset.shuffle dataset in
  let train_dataset, test_dataset = Dataset.split dataset 0.8 in

  let train_loader = Dataloader.create train_dataset ~batch_size:4 ~shuffle:true ~transforms:[Transform.normalize] in
  let test_loader = Dataloader.create test_dataset ~batch_size:10 ~shuffle:false ~transforms:[Transform.normalize] in

  let model = Model.create_Sequential [
    Model.create_Linear ~in_features:4 ~out_features:50 ~bias:true;
    Model.create_ReLU ();
    Model.create_Linear ~in_features:50 ~out_features:3 ~bias:true;
  ] in 

  let loss_func = Model.create_CrossEntropy () in
  let optimizer = Optimizer.create_SGD ~params:Model.(model.parameters) ~lr:0.01 in

  let test () =
    let correct = ref 0 in
    let total = ref 0 in
    for batch_idx = 0 to Dataloader.get_total_batches test_loader - 1 do
      let batch = Dataloader.get_batch test_loader batch_idx in
      let output = Model.forward model [batch.data] in
      (* Printf.printf "Output: %s\n" (Tensor.to_string output); *)
  
      let output_labels = Ndarray.dargmax output.data 1 in
      let target_labels = Ndarray.dargmax batch.label.data 1 in
  
      (* Calculate the number of correct predictions *)
      for i = 0 to Array.length output_labels.data - 1 do
        if int_of_float output_labels.data.(i) = int_of_float target_labels.data.(i) then
          incr correct;
        incr total;
      done;
  
      (* let loss = Model.forward loss_func [output; batch.label] in *)
      (* Printf.printf "Target: %s\n" (Tensor.to_string batch.label); *)
      (* Printf.printf "Output label: %s\n" (Ndarray.to_string output_labels); *)
      (* Printf.printf "Target label: %s\n" (Ndarray.to_string target_labels); *)
      (* Printf.printf "Test Loss: %s\n" (Tensor.to_string loss); *)
    done;
    let accuracy = (float_of_int !correct) /. (float_of_int !total) *. 100.0 in
    accuracy
  in

  let train () =
    for epoch = 1 to 20 do
      let total_loss = ref 0.0 in
      let total_batches = Dataloader.get_total_batches train_loader in
      for batch_idx = 0 to total_batches - 1 do
        let batch = Dataloader.get_batch train_loader batch_idx in
        (* Printf.printf "Input: %s\n" (Tensor.to_string batch.data); *)
        (* Printf.printf "Target: %s\n" (Tensor.to_string batch.label); *)
        
        let output = Model.forward model [batch.data] in
        (* Printf.printf "Output: %s\n" (Tensor.to_string output); *)
        let loss = Model.forward loss_func [output; batch.label] in
        (* Printf.printf "Loss: %s\n"  (Tensor.to_string loss); *)
        total_loss := !total_loss +. loss.data.data.(0);
        optimizer.zero_grad ();
        Utils.backprop loss;
        optimizer.step ()
      done;
      let avg_loss = !total_loss /. float_of_int total_batches in
      let test_acc = test () in
      Printf.printf "Epoch: %d, Average Train Loss: %.4f Test Accuracy: %.2f%%\n"  epoch avg_loss test_acc;
      flush stdout;
    done
  in
  train ()


let example_mnist () =

  (* Create datasets *)
  let train_dataset, test_dataset = Dataset.load_cnn_mnist () in

  (* Create data loaders *)
  let train_loader = Dataloader.create train_dataset ~batch_size:256 ~shuffle:true ~transforms:[Transform.normalize] in
  let test_loader = Dataloader.create test_dataset ~batch_size:1000 ~shuffle:false ~transforms:[Transform.normalize] in

  (* Define the model *)
  (* let model = Model.create_Sequential [
    Model.create_Linear ~in_features:(28 * 28) ~out_features:32 ~bias:true;
    Model.create_ReLU ();
    (* Model.create_Linear ~in_features:128 ~out_features:64 ~bias:true;
    Model.create_ReLU (); *)
    Model.create_Linear ~in_features:32 ~out_features:10 ~bias:true;
  ] in  *)
  Printf.printf "Create Model\n";
  let model = Model.create_Sequential [
    (* 第1层：卷积层 + ReLU 激活 *)
    Model.create_Conv2d ~in_channels:1 ~out_channels:32 ~kernel_size:3 ~stride:1 ~padding:1 ~bias:true;
    Model.create_ReLU ();
  
    (* 第2层：池化层 *)
    Model.create_MeanPool2d ~kernel_size:2 ~stride:2;
  
    (* 第3层：卷积层 + ReLU 激活 *)
    Model.create_Conv2d ~in_channels:32 ~out_channels:64 ~kernel_size:3 ~stride:1 ~padding:1 ~bias:true;
    Model.create_ReLU ();
  
    (* 第4层：池化层 *)
    Model.create_MeanPool2d ~kernel_size:2 ~stride:2;
  
    (* 第5层：Flatten 展平层 *)
    Model.create_Flatten ();
  
    (* 第6层：全连接层1 + ReLU *)
    Model.create_Linear ~in_features:(7 * 7 * 64) ~out_features:128 ~bias:true;
    Model.create_ReLU ();
  
    (* 第7层：全连接层2（输出层） *)
    Model.create_Linear ~in_features:128 ~out_features:10 ~bias:true;
  ] in
  
  (* Define the loss function and optimizer *)
  let loss_func = Model.create_CrossEntropy () in
  let optimizer = Optimizer.create_SGD ~params:Model.(model.parameters) ~lr:0.01 in

  (* Define the test function *)
  let test () =
    let correct = ref 0 in
    let total = ref 0 in
    for batch_idx = 0 to Dataloader.get_total_batches test_loader - 1 do
      let batch = Dataloader.get_batch test_loader batch_idx in
      let output = Model.forward model [batch.data] in
      let output_labels = Ndarray.dargmax output.data 1 in
      let target_labels = Ndarray.dargmax batch.label.data 1 in

      (* Calculate the number of correct predictions *)
      for i = 0 to Array.length output_labels.data - 1 do
        if int_of_float output_labels.data.(i) = int_of_float target_labels.data.(i) then
          incr correct;
        incr total;
      done;
    done;
    let accuracy = (float_of_int !correct) /. (float_of_int !total) *. 100.0 in
    accuracy
  in

  (* Define the train function *)
  let train () =
    for epoch = 1 to 20 do
      Printf.printf "Epoch %d/%d\n" epoch 20;
      flush stdout;
      let total_loss = ref 0.0 in
      let total_batches = Dataloader.get_total_batches train_loader in
      for batch_idx = 0 to total_batches - 1 do
        Printf.printf "Batch %d/%d\n" batch_idx total_batches;
        flush stdout;
        let batch = Dataloader.get_batch train_loader batch_idx in
        let output = Model.forward model [batch.data] in
        let loss = Model.forward loss_func [output; batch.label] in
        total_loss := !total_loss +. loss.data.data.(0);
        optimizer.zero_grad ();
        Utils.backprop loss;
        optimizer.step ()
      done;
      let avg_loss = !total_loss /. float_of_int total_batches in
      let test_acc = test () in
      Printf.printf "Epoch: %d, Average Train Loss: %.4f, Test Accuracy: %.2f%%\n" epoch avg_loss test_acc;
      flush stdout;
    done
  in

  (* Start training *)
  train ()

let () =
  (* example1 (); *)
  (* example2 (); *)
  (* example3 (); *)
  example_iris ();
  example_mnist ()
  
