type tensor = Tensor.t

(* Represents a transformation applied to a tensor. *)
type t = tensor -> tensor
(** The type for transformations, where a transformation takes an input tensor and returns a transformed tensor. *)

(* 
   Normalizes the tensor.

   Returns:
   - A transformation that normalizes the input tensor to have a mean of 0 and standard deviation of 1.
 *)
val normalize : t

(* 
   Resizes the tensor.

   Parameters:
   - `width`: The desired width of the output tensor.
   - `height`: The desired height of the output tensor.

   Returns:
   - A transformation that resizes the input tensor to the specified [width] and [height].
 *)
val resize : int -> int -> t

(* 
   Rotates the tensor.

   Parameters:
   - `angle`: The angle in degrees to rotate the tensor.

   Returns:
   - A transformation that rotates the input tensor by the specified [angle].
 *)
val rotate : float -> t

(* 
   Translates the tensor.

   Parameters:
   - `x`: The offset in the x-axis.
   - `y`: The offset in the y-axis.

   Returns:
   - A transformation that translates the input tensor by the specified [x] and [y] offsets.
 *)
val translate : float -> float -> t

(* 
   Scales the tensor.

   Parameters:
   - `x`: The scaling factor along the x-axis.
   - `y`: The scaling factor along the y-axis.

   Returns:
   - A transformation that scales the input tensor by the specified [x] and [y] factors.
 *)
val scale : float -> float -> t

(* 
   Flips the tensor.

   Parameters:
   - `horizontal`: If true, flips the tensor horizontally.
   - `vertical`: If true, flips the tensor vertically.

   Returns:
   - A transformation that flips the input tensor based on the specified [horizontal] and [vertical] options.
 *)
val flip : bool -> bool -> t