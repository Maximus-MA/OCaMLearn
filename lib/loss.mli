
type tensor = Tensor.t

(** Mean squared error loss. *)
val mse_loss : prediction:tensor -> target:tensor -> tensor

(** Cross-entropy loss. *)
val cross_entropy_loss : prediction:tensor -> target:tensor -> tensor
