type ndarray = Ndarray.t

(* Represents a dataset consisting of data and labels. *)
type t = {
  data : ndarray;  (* The data for the dataset. *)
  label : ndarray; (* The labels for the dataset. *)
}

val get_item : t -> int -> ndarray * ndarray
(** [get_item dataset idx] retrieves the sample at the specified index [idx] from the dataset [dataset].
    It returns a tuple containing the data tensor and the label tensor for the sample. *)

val shuffle : t -> t
(** [shuffle dataset] shuffles the samples in the dataset [dataset] and returns a new dataset with the samples randomly shuffled. *)

val split : t -> float -> t * t
(** [split dataset ratio] splits the dataset [dataset] into two parts based on the specified [ratio].
    The first part contains the first [ratio] fraction of samples, and the second part contains the remaining samples. *)





