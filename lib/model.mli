type tensor = Tensor.t

type t = {
    parameters: tensor list;
    forward_fn: t -> tensor list -> tensor;
}

val forward : t -> tensor list -> tensor

val get_parameters : t -> tensor list

val create_Linear : in_features:int -> out_features:int -> bias:bool -> t

val create_Conv2d : in_channels:int -> out_channels:int -> kernel_size:int -> stride:int -> padding:int -> t

val create_Flatten : unit -> t

val create_Sequential : t list -> t

val create_ReLU : unit -> t 

val create_Sigmoid : unit -> t

val create_LeakyReLU : ?alpha:float -> t

val create_Tanh : unit -> t

val create_Softmax : unit -> t

val create_MSE : unit -> t

val create_CrossEntropy : unit -> t

