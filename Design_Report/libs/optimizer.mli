(** The type representing an optimizer. *)
type tensor = Tensor.t

type t = {
  parameters: tensor list;  (* List of tensors representing the parameters to optimize. *)
  step: unit -> unit;       (* Function to update parameters based on their gradients. *)
  zero_grad: unit -> unit;  (* Function to reset all gradients to zero. *)
}

(* 
   Creates an SGD (Stochastic Gradient Descent) optimizer.

   Parameters:
   - `params`: A list of tensors to be optimized.
   - `lr`: The learning rate for the optimizer.

   Returns:
   - An instance of the SGD optimizer.
 *)
val create_SGD : params:tensor list -> lr:float -> t

(* 
   Creates an Adam optimizer.

   Parameters:
   - `params`: A list of tensors to be optimized.
   - `lr`: The learning rate for the optimizer.
   - `beta1`: The first exponential decay rate for the moment estimates.
   - `beta2`: The second exponential decay rate for the moment estimates.
   - `eps`: A small constant to prevent division by zero.

   Returns:
   - An instance of the Adam optimizer.
 *)
val create_Adam : params:tensor list -> lr:float -> beta1:float -> beta2:float -> eps:float -> t

(* 
   Performs a single optimization step.

   Parameters:
   - `optimizer`: The optimizer instance.

   Returns:
   - Updates the parameters of the optimizer in place. No return value.
 *)
val step : t -> unit

(* 
   Resets all gradients of the parameters to zero.

   Parameters:
   - `optimizer`: The optimizer instance.

   Returns:
   - Clears the gradients of all parameters. No return value.
 *)
val zero_grad : t -> unit

