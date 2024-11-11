type tensor = Tensor.t

(** The signature for a neural network module. *)
module type Model = sig
  type t

  (** Forward pass through the module. *)
  val init: t -> unit

  val forward : t -> tensor -> tensor

  (** Get the parameters of the module. *)
  val parameters : t -> tensor list
end

(** Linear (fully connected) layer. *)
module Linear : sig
  type t

  (** Create a linear layer with the given input and output sizes. *)
  val create : in_features:int -> out_features:int -> t

  (** Forward pass through the linear layer. *)
  val forward : t -> tensor -> tensor

  (** Get the parameters of the linear layer. *)
  val parameters : t -> tensor list
end

module ReLU : sig
  type t

  (** Create a ReLU layer. *)
  val create : unit -> t

  (** Forward pass through the ReLU layer. *)
  val forward : t -> tensor -> tensor

  (** Get the parameters of the ReLU layer (empty list). *)
  val parameters : t -> tensor list
end

(** Sequential container module. *)
module Sequential : sig
  type t

  (** Create a sequential module from a list of modules. *)
  val create : (module NNModule with type t = 'a) list -> t

  (** Forward pass through the sequential module. *)
  val forward : t -> tensor -> tensor

  (** Get the parameters of all modules in the sequence. *)
  val parameters : t -> tensor list
end
