(* Linear.ml *)
type tensor = Tensor.t
module Linear : Model = struct
  type parameters = {
    weights : tensor;  (* 具体字段 *)
    bias : tensor;
  }

  type t = {
    parameters: parameters;
    forward: t -> tensor list -> tensor;
    get_parameters: t -> tensor list;
  }

  let forward linear_layer input =
    Array.map2 (fun b w ->
      Array.fold_left (fun acc (x, w) -> acc +. x *. w) b (Array.combine input w)
    ) linear_layer.parameters.bias linear_layer.parameters.weights

  let get_parameters linear_layer = 
      [linear_layer.parameters.weights; linear_layer.parameters.bias]
  
  let create (n_input : int) (n_output : int) : t =
    let weights = Array.init n_output (fun _ -> Array.init n_input (fun _ -> Random.float 1.0)) in
    let bias = Array.init n_output (fun _ -> Random.float 1.0) in
  {parameters= { weights=weights; bias=bias }; forward=forward; get_parameters=parameters}



end
