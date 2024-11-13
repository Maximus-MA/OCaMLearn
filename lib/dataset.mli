type ndarray = Ndarray.t

type t = {
  data : ndarray;  (** The data for the dataset. *)
  label : ndarray;  (** The labels for the dataset. *)
}

val get_item : t -> int -> ndarray*ndarray
(** [get_item dataset idx] returns the sample at index [idx] from the dataset. *)

val shuffle : t -> t
(** [shuffle dataset] returns a new dataset with samples randomly shuffled. *)

val split : t -> float -> t * t
(** [split dataset ratio] splits the dataset into two parts based on the given ratio.
    - The first dataset contains the first [ratio] fraction of samples.
    - The second dataset contains the remaining samples. *)
