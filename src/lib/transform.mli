type ndarray = Ndarray.t
(* Represents a transformation applied to a ndarray. *)
type t = ndarray -> ndarray
(** The type for transformations, where a transformation takes an input ndarray and returns a transformed ndarray. *)

val normalize : t
(** [normalize] normalizes the input ndarray to have a mean of 0 and standard deviation of 1.
    It returns a transformation that can be applied to the ndarray. *)

val resize : int -> int -> t
(** [resize width height] resizes the input ndarray to the specified [width] and [height].
    It returns a transformation that resizes the ndarray accordingly. *)

val rotate : float -> t
(** [rotate angle] rotates the input ndarray by the specified [angle] in degrees.
    It returns a transformation that applies the rotation to the ndarray. *)

val translate : float -> float -> t
(** [translate x y] translates the input ndarray by the specified [x] and [y] offsets along the x-axis and y-axis respectively.
    It returns a transformation that applies the translation to the ndarray. *)

val scale : float -> float -> t
(** [scale x y] scales the input ndarray by the specified [x] and [y] scaling factors along the x-axis and y-axis respectively.
    It returns a transformation that scales the ndarray accordingly. *)

val flip : bool -> bool -> t
(** [flip horizontal vertical] flips the input ndarray based on the specified [horizontal] and [vertical] options.
    If [horizontal] is true, the ndarray is flipped horizontally; if [vertical] is true, the ndarray is flipped vertically. 
    It returns a transformation that applies the flip operation. *)
