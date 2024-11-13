type tensor = Tensor.t

type parameters

type t = {
    parameters: parameters;
    forward: t -> tensor list -> tensor;
}

val forward : t -> tensor list -> tensor;

val create

val parameters : t -> tensor list




