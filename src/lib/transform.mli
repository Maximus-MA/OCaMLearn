type ndarray = Ndarray.t
(* Represents a transformation applied to a ndarray. *)
type t = ndarray -> ndarray
(** The type for transformations, where a transformation takes an input ndarray and returns a transformed ndarray. *)

val normalize : t
(** [normalize] normalizes the input ndarray to have a mean of 0 and standard deviation of 1.
    It returns a transformation that can be applied to the ndarray. *)

val image_scale : t
(** [image_scale t] scales the image to [-1,1] *)
