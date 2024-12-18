type tensor = Tensor.t

(* Represents a neural network layer or module. *)
type t = {
  parameters: tensor list;          (* List of tensors representing the layer's parameters. *)
  forward_fn: tensor list -> tensor; (* Function to compute the forward pass of the layer. *)
}

val forward : t -> tensor list -> tensor
(** [forward layer inputs] performs a forward pass through the given layer [layer] using the provided list of input tensors [inputs].
    It returns the output tensor resulting from the forward pass. *)

val get_parameters : t -> tensor list
(** [get_parameters layer] retrieves the parameters of the given layer [layer].
    It returns a list of tensors representing the layer's parameters. *)

val create_Linear : in_features:int -> out_features:int -> bias:bool -> t
(** [create_Linear in_features out_features bias] creates a new linear (fully connected) layer with the specified number of input features [in_features],
    output features [out_features], and an optional bias term [bias]. *)

val create_Conv2d : 
    in_channels:int -> 
    out_channels:int -> 
    kernel_size:int -> 
    stride:int -> 
    padding:int -> 
    bias:bool ->
    t
(** [create_Conv2d in_channels out_channels kernel_size stride padding] creates a new 2D convolutional layer with the specified number of input channels [in_channels],
    output channels [out_channels], kernel size [kernel_size], stride [stride], and padding [padding]. *)

val create_MeanPool2d : 
kernel_size:int -> 
stride:int -> 
t
(** [create_MeanPool2d kernel_size stride] creates a 2D mean pooling layer with the specified pooling window size [kernel_size]
    and stride [stride]. The layer performs average pooling over the input tensor to reduce its spatial dimensions. *)

      
val create_Flatten : unit -> t
(** [create_Flatten ()] creates a layer that flattens the input tensor into a 1D vector. *)

val create_Sequential : t list -> t
(** [create_Sequential layers] creates a sequential container that chains together the given list of layers [layers]. *)

val create_ReLU : unit -> t
(** [create_ReLU ()] creates a ReLU activation layer that applies the ReLU activation function element-wise. *)

val create_Sigmoid : unit -> t
(** [create_Sigmoid ()] creates a Sigmoid activation layer that applies the Sigmoid activation function element-wise. *)

val create_LeakyReLU : ?alpha:float -> t
(** [create_LeakyReLU ?alpha] creates a Leaky ReLU activation layer that applies the Leaky ReLU activation function element-wise.
    Optionally, a negative slope [alpha] can be specified. *)

val create_Tanh : unit -> t
(** [create_Tanh ()] creates a Tanh activation layer that applies the Tanh activation function element-wise. *)

val create_Softmax : unit -> t
(** [create_Softmax ()] creates a Softmax activation layer that applies the Softmax function along the last dimension of the input. *)

val create_MSE : unit -> t
(** [create_MSE ()] creates a Mean Squared Error (MSE) loss layer that computes the MSE loss between predictions and true labels. *)

val create_CrossEntropy : unit -> t
(** [create_CrossEntropy ()] creates a Cross Entropy loss layer that computes the Cross Entropy loss for classification tasks. *)
