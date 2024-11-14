(* Open necessary modules *)
open Dataset
open Data_loader
open Model
open Optimizer
open Tensor
open Transform

(* Assume we have a function to load the MNIST dataset into ndarrays *)
let load_mnist () : Dataset.t =
  (* Mock function to load MNIST data *)
  let num_samples = 60000 in
  let num_features = 28 * 28 in
  let num_classes = 10 in

  (* Create random data to simulate MNIST images and labels *)
  let train_images = Ndarray.rand [| num_samples; num_features |] in
  let train_labels = Ndarray.rand [| num_samples |] in
  { data = train_images; label = train_labels }

(* Load the dataset *)
let dataset = load_mnist ()

(* Shuffle and split the dataset *)
let dataset = Dataset.shuffle dataset
let train_dataset, val_dataset = Dataset.split dataset 0.8

(* Define transformations (if any) *)
let transforms = [ Transform.normalize ]

(* Create data loaders *)
let batch_size = 64
let train_loader = Data_loader.create ~dataset:train_dataset ~batch_size ~shuffle:true ~transorms:transforms
let val_loader = Data_loader.create ~dataset:val_dataset ~batch_size ~shuffle:false ~transorms:transforms

(* Define the model *)
let model = Model.create_Sequential [
  Model.create_Linear ~in_features:784 ~out_features:128 ~bias:true;
  Model.create_ReLU ();
  Model.create_Linear ~in_features:128 ~out_features:64 ~bias:true;
  Model.create_ReLU ();
  Model.create_Linear ~in_features:64 ~out_features:10 ~bias:true;
  Model.create_Softmax ();
]

(* Define the loss function *)
let criterion = Model.create_CrossEntropy ()

(* Get model parameters *)
let parameters = Model.get_parameters model

(* Create an optimizer *)
let optimizer = Optimizer.create_SGD ~params:parameters ~lr:0.01

(* Training loop *)
let num_epochs = 10

for epoch = 1 to num_epochs do
  Printf.printf "Epoch %d/%d\n%!" epoch num_epochs;

  let total_batches = Data_loader.get_total_batches train_loader in

  for batch_idx = 0 to total_batches - 1 do
    (* Get a batch of data *)
    let inputs, labels = Data_loader.get_batch train_loader batch_idx in

    (* Forward pass *)
    let outputs = Model.forward model [ inputs ] in

    (* Compute loss *)
    let loss = Model.forward criterion [ outputs; labels ] in

    (* Backward pass *)
    Utils.backprop loss;

    (* Update parameters *)
    optimizer.step ();

    (* Zero the gradients *)
    optimizer.zero_grad ();

    (* Print training progress *)
    if batch_idx mod 100 = 0 then
      Printf.printf "Batch %d/%d, Loss: %f\n%!" batch_idx total_batches (Tensor.get_data loss).(0)
  done;

  (* Validation phase *)
  let val_batches = Data_loader.get_total_batches val_loader in
  let total_loss = ref 0.0 in
  let correct = ref 0 in
  let total = ref 0 in

  for batch_idx = 0 to val_batches - 1 do
    let inputs, labels = Data_loader.get_batch val_loader batch_idx in

    (* Forward pass *)
    let outputs = Model.forward model [ inputs ] in

    (* Compute loss *)
    let loss = Model.forward criterion [ outputs; labels ] in
    total_loss := !total_loss +. (Tensor.get_data loss).(0);

    (* Calculate accuracy *)
    let predictions = Tensor.dargmax outputs 1 in
    let correct_preds = Tensor.eq predictions labels in
    let batch_correct = Tensor.sum correct_preds |> int_of_float in
    correct := !correct + batch_correct;
    total := !total + (Tensor.shape labels).(0);
  done;

  let avg_loss = !total_loss /. float_of_int val_batches in
  let accuracy = (float_of_int !correct /. float_of_int !total) *. 100.0 in
  Printf.printf "Validation Loss: %f, Accuracy: %.2f%%\n%!" avg_loss accuracy;
done
