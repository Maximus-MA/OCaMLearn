# OCaml Deep Learning Framework
Currently, for the low-level `Ndarray` module, we have completed the implementation and testing of most functions. For the `Tensor` module, we have implemented and tested several key functions, especially those required for building a simple MLP network. For the `Optimizer` module, we have implemented the basic SGD optimizer. For the `Model` module, we have implemented functionalities related to linear layers, ReLU activation functions, cross-entropy loss, and model construction using `create_Sequential`. Additionally, we have implemented the gradient backpropagation functionality in the `Until` module.

For the `Dataset` and `Dataloader` modules, we have completed the implementation of basic features such as creating datasets and loading data in batches. In `main.ml`, we demonstrate the process of constructing a simple MLP network with one hidden layer and performing forward and backward propagation using randomly generated input data.

As for future plans, we will continue to work on the implementation and testing of the `Tensor`, `Model`, `Dataset`, `Dataloader`, `Optimizer`, and `Transform` modules. This includes adding convolution operations, loading real datasets, performing data augmentation, and implementing more complex optimizers such as Adam.

## High-level Example
Here is an example to use our framework to train a MLP on random data.
```ocaml
let input = Tensor.ones [|2; 100|] in
  let model = Model.create_Sequential [
    Model.create_Linear ~in_features:100 ~out_features:50 ~bias:true;
    Model.create_ReLU ();
    Model.create_Linear ~in_features:50 ~out_features:10 ~bias:true;
    ] in 
  let loss_func = Model.create_CrossEntropy () in
  let target =  Tensor.zeros [|2; 10|] in
  Tensor.set target [|0; 0|] 1.;
  Tensor.set target [|1; 2|] 1.;
  let optimizer = Optimizer.create_SGD ~params:Model.(model.parameters) ~lr:0.01 in
  for _ = 0 to 10 do
    let output = Model.forward model [input] in
    let loss = Model.forward loss_func [output; target] in
    Printf.printf "Loss: %s\n" (Tensor.to_string loss);
    optimizer.zero_grad ();
    Utils.backprop loss;
    optimizer.step ()
  done
```
output
```
Loss: Tensor {data = 38.235257119, requires_grad = true}
Loss: Tensor {data = 32.6014171402, requires_grad = true}
Loss: Tensor {data = 27.6126259998, requires_grad = true}
Loss: Tensor {data = 22.8692291298, requires_grad = true}
Loss: Tensor {data = 18.2847880817, requires_grad = true}
Loss: Tensor {data = 13.7752143721, requires_grad = true}
Loss: Tensor {data = 9.25994272481, requires_grad = true}
Loss: Tensor {data = 4.88796455224, requires_grad = true}
Loss: Tensor {data = 3.04297275696, requires_grad = true}
Loss: Tensor {data = 0.733882587709, requires_grad = true}
Loss: Tensor {data = 0.836976953813, requires_grad = true}
```
According to output, we can see that the model is converging!