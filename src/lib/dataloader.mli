type dataset = Dataset.t
type tensor = Tensor.t
type transform = Transform.t

(* Represents a dataset consisting of data and labels. *)
type tensor_dataset = {
  data : tensor; (* Data tensor containing the samples. *)
  label : tensor; (* Label tensor corresponding to the data. *)
}

(* DataLoader type containing the dataset and batching information. *)
type t = {
  dataset : tensor_dataset; (* Tensor dataset used for loading batches. *)
  batch_size : int; (* Number of samples per batch. *)
  total_batches : int; (* Total number of batches available. *)
}

val create : ?transforms:transform list -> dataset -> batch_size:int -> shuffle:bool -> t
(** [create ?transforms dataset batch_size shuffle] creates a new data loader
    from the specified dataset. Optionally applies a list of transformations to
    the dataset. If [shuffle] is true, the dataset is shuffled before batching.*)

val get_batch : t -> int -> tensor_dataset
(** [get_batch dataloader idx] retrieves the batch at the specified index [idx]
    from the data loader [dataloader]. *)

val get_total_batches : t -> int
(** [get_total_batches dataloader] returns the total number of batches available
    in the data loader [dataloader]. *)
