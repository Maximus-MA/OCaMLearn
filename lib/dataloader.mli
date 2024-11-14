type dataset = Dataset.t

type tensor = Tensor.t

type transform = Transform.t

type tensor_dataset = {
    data : tensor;
    label : tensor;
}

type t = {
  dataset : tensor_dataset;          (** The dataset to load from. *)
  batch_size : int;             (** Number of samples per batch. *)
  total_batches : int;          (** Total number of batches in the dataset. *)
}

val create : dataset -> batch_size:int -> shuffle:bool -> ?transorms: transform list -> t
(** [create dataset batch_size shuffle] creates a data loader with the specified [dataset],
    [batch_size], and [shuffle] option.
*)

val get_batch : t -> int -> tensor*tensor
(** [get_batch loader idx] returns the batch at index [idx] from the loader. *)

val get_total_batches : t -> int
(** [get_total_batches loader] returns the total number of batches in the loader. *)
