type tensor = Tensor.t
(** Represents a tensor in the computational graph. *)

val backprop : tensor -> unit
(** [backprop tensor] performs backpropagation starting from the given [tensor] through the entire computational graph. 
    It first performs a topological sort of the graph starting from the input tensor, then sequentially calls the backward_fn 
    of all parameters in the correct order to compute gradients.
    It updates the gradients of all tensors in the computational graph connected to the [tensor], with no return value. *)

