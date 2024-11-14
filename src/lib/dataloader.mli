type dataset = Dataset.t

type tensor = Tensor.t

type transform = Transform.t

(* Represents a dataset consisting of data and labels. *)
type tensor_dataset = {
    data : tensor;  (* Data tensor containing the samples. *)
    label : tensor; (* Label tensor corresponding to the data. *)
}

(* DataLoader type containing the dataset and batching information. *)
type t = {
  dataset : tensor_dataset;  (* Tensor dataset used for loading batches. *)
  batch_size : int;          (* Number of samples per batch. *)
  total_batches : int;       (* Total number of batches available. *)
}

(* 
   Creates a new data loader.

   Parameters:
   - `dataset`: The input dataset to load batches from.
   - `batch_size`: The number of samples in each batch.
   - `shuffle`: Whether to shuffle the dataset before batching.
   - `transorms`: Optional list of transformations to apply to the dataset.

   Returns:
   - A data loader configured with the specified parameters.
 *)
val create : dataset -> batch_size:int -> shuffle:bool -> ?transorms: transform list -> t

(* 
   Retrieves a batch from the data loader.

   Parameters:
   - `dataloader`: The data loader instance.
   - `idx`: The batch index to retrieve.

   Returns:
   - A tuple containing:
     - The data tensor for the batch.
     - The label tensor for the batch.
 *)
val get_batch : t -> int -> tensor * tensor

(* 
   Gets the total number of batches in the data loader.

   Parameters:
   - `dataloader`: The data loader instance.

   Returns:
   - The total number of batches available in the data loader.
 *)
val get_total_batches : t -> int