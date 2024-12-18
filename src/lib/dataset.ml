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
  if ratio <= 0.0 || ratio >= 1.0 then
    invalid_arg "Ratio must be between 0.0 and 1.0!";

  let num_samples = dataset.data.shape.(0) in
  let split_idx = int_of_float (float_of_int num_samples *. ratio) in

  (* Generate slicing ranges for the split *)
  let generate_slice_ranges shape start_point end_point =
    Array.to_list (Array.mapi (fun i dim_size ->
        if i = 0 then (start_point, end_point) else (0, dim_size)
      ) shape)
  in

  let data_train_ranges = generate_slice_ranges dataset.data.shape 0 split_idx in
  let data_test_ranges = generate_slice_ranges dataset.data.shape split_idx num_samples in
  let label_train_ranges = generate_slice_ranges dataset.label.shape 0 split_idx in
  let label_test_ranges = generate_slice_ranges dataset.label.shape split_idx num_samples in

  let data_train = Ndarray.slice dataset.data data_train_ranges in
  let data_test = Ndarray.slice dataset.data data_test_ranges in
  let label_train = Ndarray.slice dataset.label label_train_ranges in
  let label_test = Ndarray.slice dataset.label label_test_ranges in

  (
    { data = data_train; label = label_train },
    { data = data_test; label = label_test }
  )
  
(* let label_string_to_float label_string = 
match label_string with
| "class_0" -> 0.0
| "class_1" -> 1.0
| _ -> failwith "Unknown label" *)

let one_hot_encode num_classes label =
  let one_hot = Array.make num_classes 0.0 in
  one_hot.(label) <- 1.0;
  one_hot

let load_csv file_path label_col label_string_to_int num_classes =
  let csv_data = Csv.load file_path in
  let num_samples = List.length csv_data in
  let feature_dim = (List.hd csv_data |> List.length) - 1 in
  let features, labels =
    List.fold_left (fun (feat_acc, label_acc) row ->
      let features = List.filteri (fun i _ -> i <> label_col) row 
                     |> List.map float_of_string 
                     |> Array.of_list in
      let label = List.nth row label_col |> label_string_to_int |> one_hot_encode num_classes in
      (Array.append feat_acc features, Array.append label_acc label)
    ) (Array.make 0 0.0, Array.make 0 0.0) csv_data
  in
  let features_ndarray = Ndarray.create features [| num_samples ; feature_dim |] in
  let labels_tensor = Ndarray.create labels [| num_samples; num_classes |] in
  { data = features_ndarray; label = labels_tensor }

let read_int ic =
  let b1 = input_byte ic in
  let b2 = input_byte ic in
  let b3 = input_byte ic in
  let b4 = input_byte ic in
  (b1 lsl 24) lor (b2 lsl 16) lor (b3 lsl 8) lor b4

let read_images filename =
  let ic = open_in_bin filename in
  let magic_number = read_int ic in
  if magic_number <> 2051 then failwith "Invalid magic number for image file";
  let num_images = read_int ic in
  let num_rows = read_int ic in
  let num_cols = read_int ic in
  let images = Array.init num_images (fun _ ->
    Array.init (num_rows * num_cols) (fun _ ->
      input_byte ic |> float_of_int
    )
  ) in
  close_in ic;
  images

let read_labels filename =
  let ic = open_in_bin filename in
  let magic_number = read_int ic in
  if magic_number <> 2049 then failwith "Invalid magic number for label file";
  let num_labels = read_int ic in
  let labels = Array.init num_labels (fun _ ->
    input_byte ic |> float_of_int
  ) in
  close_in ic;
  labels


let convert_labels_to_one_hot labels num_classes =
  Array.map (fun label -> one_hot_encode num_classes (int_of_float label)) labels

let load_cnn_mnist () =
  (* Load the MNIST dataset *)
  let train_images = read_images "dataset/mnist/train-images-idx3-ubyte" in
  let train_labels = read_labels "dataset/mnist/train-labels-idx1-ubyte" in
  let test_images = read_images "dataset/mnist/t10k-images-idx3-ubyte" in
  let test_labels = read_labels "dataset/mnist/t10k-labels-idx1-ubyte" in

  (* Convert the labels to one-hot encoding *)
  let train_labels = convert_labels_to_one_hot train_labels 10 in
  let test_labels = convert_labels_to_one_hot test_labels 10 in

  (* Calculate reduced size: 10% of the original training set *)
  let original_size = Array.length train_images in
  let target_size = max 1 (original_size) in  (* Ensure at least 1 sample *)

  (* Slice the first 10% of train_images and train_labels *)
  let reduced_train_images = Array.sub train_images 0 target_size in
  let reduced_train_labels = Array.sub train_labels 0 target_size in

  (* Reshape the training images for CNN input: [batch_size, 1, 28, 28] *)
  let reshape_images images =
    Array.map (fun image ->
      Array.init 1 (fun _ -> image)  (* Add a channel dimension *)
    ) images
  in
  let cnn_train_images = reshape_images reduced_train_images in
  let cnn_test_images = reshape_images test_images in

  (* Convert the reshaped data to ndarrays *)
  let train_data = Ndarray.create 
      (Array.concat (Array.to_list (Array.concat (Array.to_list cnn_train_images))))
      [| target_size; 1; 28; 28 |] in

  let train_labels = Ndarray.create 
      (Array.concat (Array.to_list reduced_train_labels)) 
      [| target_size; 10 |] in

  let test_data = Ndarray.create 
      (Array.concat (Array.to_list (Array.concat (Array.to_list cnn_test_images))))
      [| Array.length test_images; 1; 28; 28 |] in

  let test_labels = Ndarray.create 
      (Array.concat (Array.to_list test_labels)) 
      [| Array.length test_labels; 10 |] in

  (* Create datasets *)
  let train_dataset = { data = train_data; label = train_labels } in
  let test_dataset = { data = test_data; label = test_labels } in
  (train_dataset, test_dataset)


let load_mnist_full () =
  (* Load the MNIST dataset *)
  let train_images = read_images "dataset/mnist/train-images-idx3-ubyte" in
  let train_labels = read_labels "dataset/mnist/train-labels-idx1-ubyte" in
  let test_images = read_images "dataset/mnist/t10k-images-idx3-ubyte" in
  let test_labels = read_labels "dataset/mnist/t10k-labels-idx1-ubyte" in

  (* Convert the labels to one-hot encoding *)
  let train_labels = convert_labels_to_one_hot train_labels 10 in
  let test_labels = convert_labels_to_one_hot test_labels 10 in

  (* Convert the data to ndarrays *)
  let train_data = Ndarray.create (Array.concat (Array.to_list train_images)) [| Array.length train_images; 28 * 28 |] in
  let train_labels = Ndarray.create (Array.concat (Array.to_list train_labels)) [| Array.length train_labels; 10 |] in
  let test_data = Ndarray.create (Array.concat (Array.to_list test_images)) [| Array.length test_images; 28 * 28 |] in
  let test_labels = Ndarray.create (Array.concat (Array.to_list test_labels)) [| Array.length test_labels; 10 |] in

  (* Create datasets *)
  let train_dataset = { data = train_data; label = train_labels } in
  let test_dataset = { data = test_data; label = test_labels } in
  (* let train_dataset = test_dataset in *)
  (train_dataset, test_dataset)
  

let load_mnist_small () =
  (* Load the MNIST dataset *)
  let train_images = read_images "dataset/mnist/train-images-idx3-ubyte" in
  let train_labels = read_labels "dataset/mnist/train-labels-idx1-ubyte" in
  let test_images = read_images "dataset/mnist/t10k-images-idx3-ubyte" in
  let test_labels = read_labels "dataset/mnist/t10k-labels-idx1-ubyte" in

  (* Convert the labels to one-hot encoding *)
  let train_labels = convert_labels_to_one_hot train_labels 10 in
  let test_labels = convert_labels_to_one_hot test_labels 10 in

  (* Calculate reduced size: 10% of the original training set *)
  let original_size = Array.length train_images in
  let target_size = max 1 (original_size / 10) in  (* Ensure at least 1 sample *)

  (* Slice the first 10% of train_images and train_labels *)
  let reduced_train_images = Array.sub train_images 0 target_size in
  let reduced_train_labels = Array.sub train_labels 0 target_size in

  (* Convert the reduced data to ndarrays *)
  let train_data = Ndarray.create 
      (Array.concat (Array.to_list reduced_train_images)) 
      [| target_size; 28 * 28 |] in

  let train_labels = Ndarray.create 
      (Array.concat (Array.to_list reduced_train_labels)) 
      [| target_size; 10 |] in

  (* Convert the test data to ndarrays (unchanged) *)
  let test_data = Ndarray.create 
      (Array.concat (Array.to_list test_images)) 
      [| Array.length test_images; 28 * 28 |] in

  let test_labels = Ndarray.create 
      (Array.concat (Array.to_list test_labels)) 
      [| Array.length test_labels; 10 |] in

  (* Create datasets *)
  let train_dataset = { data = train_data; label = train_labels } in
  let test_dataset = { data = test_data; label = test_labels } in
  (train_dataset, test_dataset)

let load_mnist full_size = 
  if full_size then load_mnist_full ()
  else load_mnist_small()