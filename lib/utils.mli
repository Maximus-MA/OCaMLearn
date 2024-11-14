type tensor = Tensor.t
(** Represents a tensor in the computational graph. *)

(* 
   Performs backpropagation from a tensor to the whole computational graph.
   It will first conduct a topological sort of the computational graph starting from the input tensor,
    and then call the backward_fn of all parameters in order.

   Parameters:
   - `tensor`: The target tensor for which gradients are computed.

   Returns:
   - No return value. Updates the gradients of all tensors in the computational graph connected to [tensor].
 *)
val backprop : tensor -> unit