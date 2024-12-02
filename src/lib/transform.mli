type ndarray = Ndarray.t
(* Represents a transformation applied to a ndarray. *)
type t = ndarray -> ndarray
(** The type for transformations, where a transformation takes an input ndarray and returns a transformed ndarray. *)

(* 
   Normalizes the ndarray.

   Returns:
   - A transformation that normalizes the input ndarray to have a mean of 0 and standard deviation of 1.
 *)
val normalize : t

(* 
   Resizes the ndarray.

   Parameters:
   - `width`: The desired width of the output ndarray.
   - `height`: The desired height of the output ndarray.

   Returns:
   - A transformation that resizes the input ndarray to the specified [width] and [height].
 *)
val resize : int -> int -> t

(* 
   Rotates the ndarray.

   Parameters:
   - `angle`: The angle in degrees to rotate the ndarray.

   Returns:
   - A transformation that rotates the input ndarray by the specified [angle].
 *)
val rotate : float -> t

(* 
   Translates the ndarray.

   Parameters:
   - `x`: The offset in the x-axis.
   - `y`: The offset in the y-axis.

   Returns:
   - A transformation that translates the input ndarray by the specified [x] and [y] offsets.
 *)
val translate : float -> float -> t

(* 
   Scales the ndarray.

   Parameters:
   - `x`: The scaling factor along the x-axis.
   - `y`: The scaling factor along the y-axis.

   Returns:
   - A transformation that scales the input ndarray by the specified [x] and [y] factors.
 *)
val scale : float -> float -> t

(* 
   Flips the ndarray.

   Parameters:
   - `horizontal`: If true, flips the ndarray horizontally.
   - `vertical`: If true, flips the ndarray vertically.

   Returns:
   - A transformation that flips the input ndarray based on the specified [horizontal] and [vertical] options.
 *)
val flip : bool -> bool -> t