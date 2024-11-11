

type tensor = Tensor.t

(** The signature for a neural network module. *)
module type Model = sig
  type t = {
	forward: t -> tensor list -> tensor;
	parameters: tensor list;
  }

  val forward : t -> tensor list -> tensor

  val linear : in_features:int -> out_features:int -> t

  val squential : in_models:t list -> t

  val relu : unit -> t

  val mse_loss : unit -> t

end

