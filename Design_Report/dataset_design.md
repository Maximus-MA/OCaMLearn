# Dataset Design



## 1. Overview

### `Dataset` Module

The `Dataset` module is designed to process and manipulate data samples and their corresponding labels. It provides basic functions such as data retrieval, shuffling and splitting, which are common requirements in data preprocessing workflows.

### Dataset Structure

```ocaml
type t = {
  data : ndarray;   (* The data for the dataset. *)
  label : ndarray;  (* The labels for the dataset. *)
}

```



In the `Dataset` structure:

- **`data`**: Stores the data samples in the form of an `ndarray`.
- **`label`**: Stores the corresponding labels for each data sample in the dataset.

This structure ensures that each data sample is directly associated with a label, allowing for efficient data access and manipulation.

### Key Functionalities

The `Dataset` module provides the following core functionalities:

- **`get_item`**: Retrieves a specific data sample and its label by index.
- **`shuffle`**: Returns a new dataset with randomly shuffled samples.
- **`split`**: Divides the dataset into two parts based on a specified ratio, often used to create training and validation sets.



## 2. Mock Use

The following will show how to use the `Dataset` and `Dataloader` modules.

### Example Usage

```ocaml
(* Step 1: Initialize the Dataset *)
let data = Ndarray.of_array [|[|1.0; 2.0|]; [|3.0; 4.0|]; [|5.0; 6.0|]|] in
let label = Ndarray.of_array [|1; 0; 1|] in
let dataset = { data; label }  
(* Creating the dataset *)

(* Step 2: Shuffle the Dataset *)
let shuffled_dataset = shuffle dataset  
(* Shuffling the dataset to randomize the order *)

(* Step 3: Split the Dataset *)
let train_set, val_set = split dataset 0.8  
(* Splitting the dataset into 80% training and 20% validation *)

(* Step 4: Initialize the Dataloader *)
let batch_size = 2 in
let dataloader = Dataloader.create ~dataset:train_set ~batch_size ~shuffle:true  
(* Create the dataloader with batching *)

(* Step 5: Retrieve Batches *)
for i = 0 to Dataloader.get_total_batches dataloader - 1 do
  let batch_data, batch_label = Dataloader.get_batch dataloader i in
  (* Here, you can use batch_data and batch_label for model training or evaluation *)
done;

```



In this example:

1. **Dataset Creation**: We initialize a dataset with `data` and `label` values.
2. **Shuffle**: The dataset is shuffled to randomize the data order, which helps prevent the model from learning order-based patterns.
3. **Split**: The dataset is split into training and validation sets based on a specified ratio.
4. **Dataloader Initialization**: We create a `Dataloader` instance with the training set, specifying a `batch_size` and enabling shuffling.
5. **Batch Retrieval**: We loop through the batches, retrieving each batch's data and label for processing. This setup enables the model to train in mini-batches, a common practice in machine learning.



## 3. Library Dependencies

The `Dataset` module uses the following libraries:

- **Ndarray**: Manages the multidimensional data structures used for data samples and labels.
- **Stdlib**: Provides standard array handling, randomization, and other essential utilities.



## 4. Implementation Plan

We have defined the interfaces for Dataset and Dataloader, next, we are about to write the corresponding implementations. These implementations are relatively simple with the code that Ndarray has already done, and **Rui Wang** will be working on them next week.

