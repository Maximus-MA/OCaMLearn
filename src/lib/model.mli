type tensor = Tensor.t

(* Represents a neural network layer or module. *)
type t = {
  parameters: tensor list;          (* List of tensors representing the layer's parameters. *)
  forward_fn: tensor list -> tensor; (* Function to compute the forward pass of the layer. *)
}

(* 
   Performs a forward pass through the layer.

   Parameters:
   - `layer`: The layer instance to perform the forward pass.
   - `inputs`: A list of input tensors to pass through the layer.

   Returns:
   - The output tensor resulting from the forward pass.
 *)
val forward : t -> tensor list -> tensor

(* 
   Retrieves the parameters of the layer.

   Parameters:
   - `layer`: The layer instance to get parameters from.

   Returns:
   - A list of tensors representing the layer's parameters.
 *)
val get_parameters : t -> tensor list

(* 
   Creates a linear (fully connected) layer.

   Parameters:
   - `in_features`: The number of input features.
   - `out_features`: The number of output features.
   - `bias`: Whether to include a bias term.

   Returns:
   - A new linear layer instance.
 *)
val create_Linear : in_features:int -> out_features:int -> bias:bool -> t

(* 
   Creates a 2D convolutional layer.

   Parameters:
   - `in_channels`: The number of input channels.
   - `out_channels`: The number of output channels.
   - `kernel_size`: The size of the convolutional kernel.
   - `stride`: The stride for the convolution.
   - `padding`: The padding to apply around the input.

   Returns:
   - A new 2D convolutional layer instance.
 *)
val create_Conv2d : 
  in_channels:int -> 
  out_channels:int -> 
  kernel_size:int -> 
  stride:int -> 
  padding:int -> 
  t

(* 
   Creates a flattening layer.

   Parameters:
   - `unit`: No parameters required.

   Returns:
   - A layer that flattens the input tensor.
 *)
val create_Flatten : unit -> t

(* 
   Creates a sequential container for layers.

   Parameters:
   - `layers`: A list of layers to chain together sequentially.

   Returns:
   - A sequential container representing the stacked layers.
 *)
val create_Sequential : t list -> t

(* 
   Creates a ReLU activation layer.

   Parameters:
   - `unit`: No parameters required.

   Returns:
   - A layer applying the ReLU activation function element-wise.
 *)
val create_ReLU : unit -> t

(* 
   Creates a Sigmoid activation layer.

   Parameters:
   - `unit`: No parameters required.

   Returns:
   - A layer applying the Sigmoid activation function element-wise.
 *)
val create_Sigmoid : unit -> t

(* 
   Creates a Leaky ReLU activation layer.

   Parameters:
   - `alpha`: Optional negative slope for the Leaky ReLU.

   Returns:
   - A layer applying the Leaky ReLU activation function element-wise.
 *)
val create_LeakyReLU : ?alpha:float -> t

(* 
   Creates a Tanh activation layer.

   Parameters:
   - `unit`: No parameters required.

   Returns:
   - A layer applying the Tanh activation function element-wise.
 *)
val create_Tanh : unit -> t

(* 
   Creates a Softmax activation layer.

   Parameters:
   - `unit`: No parameters required.

   Returns:
   - A layer applying the Softmax function along the last dimension.
 *)
val create_Softmax : unit -> t

(* 
   Creates a Mean Squared Error (MSE) loss layer.

   Parameters:
   - `unit`: No parameters required.

   Returns:
   - A layer computing the Mean Squared Error loss.
 *)
val create_MSE : unit -> t

(* 
   Creates a Cross Entropy loss layer.

   Parameters:
   - `unit`: No parameters required.

   Returns:
   - A layer computing the Cross Entropy loss.
 *)
val create_CrossEntropy : unit -> t