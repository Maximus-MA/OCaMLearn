type ndarray = Ndarray.t

(* Represents a dataset consisting of data and labels. *)
type t = {
  data : ndarray;  (* The data for the dataset. *)
  label : ndarray; (* The labels for the dataset. *)
}

(* 
   Retrieves a sample from the dataset.

   Parameters:
   - `dataset`: The dataset instance to retrieve the sample from.
   - `idx`: The index of the sample to retrieve.

   Returns:
   - A tuple containing:
     - The data tensor for the sample.
     - The label tensor for the sample.
 *)
val get_item : t -> int -> ndarray * ndarray

(* 
   Shuffles the dataset.

   Parameters:
   - `dataset`: The dataset instance to shuffle.

   Returns:
   - A new dataset where the samples have been randomly shuffled.
 *)
val shuffle : t -> t

(* 
   Splits the dataset into two parts.

   Parameters:
   - `dataset`: The dataset instance to split.
   - `ratio`: A float specifying the fraction of the dataset to include in the first part.

   Returns:
   - A tuple containing:
     - The first dataset, containing the first `ratio` fraction of samples.
     - The second dataset, containing the remaining samples.
 *)
val split : t -> float -> t * t