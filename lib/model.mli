type tensor = Tensor.t

type parameters

type t = {
    parameters: parameters;
    forward: t -> tensor list -> tensor;
    get_parameters: t -> tensor list;
}

val forward : t -> tensor list -> tensor;

val get_parameters : t -> tensor list

val create






