(* src/ndarray.ml *)
[@@@ocaml.warning "-27"]

type t = {
  data: float array;
  shape: int array;
}

let not_implemented feature_name =
  failwith (feature_name ^ " is not yet implemented")

let numel shape = not_implemented "numel"

let strides shape = not_implemented "strides"

let sum_multiplied a b = not_implemented "sum_multiplied"

let is_broadcastable shape1 shape2 = not_implemented "is_broadcastable"

let broadcast_shape shape1 shape2 = not_implemented "broadcast_shape"


let add a b = not_implemented "add"

let sub a b = not_implemented "sub"

let mul a b = not_implemented "mul"

let div a b = not_implemented "div"

let matmul a b = not_implemented "matmul"

let create data shape = not_implemented "create"

let create_float value = not_implemented "create_float"

let create_int value = not_implemented "create_int"

let zeros shape = not_implemented "zeros"

let ones shape = not_implemented "ones"

let rand shape = not_implemented "rand"

let xavier_init shape = not_implemented "xavier_init"

let kaiming_init shape = not_implemented "kaiming_init"

let shape arr = not_implemented "shape"

let dim arr = not_implemented "dim"

let at arr indices = not_implemented "at"

let reshape arr new_shape = not_implemented "reshape"

let set arr idx value = not_implemented "set"

let transpose arr = not_implemented "transpose"

let to_array arr = not_implemented "to_array"

let slice arr ranges = not_implemented "slice"

let fill arr value = not_implemented "fill"

let sum arr = not_implemented "sum"

let mean arr = not_implemented "mean"

let var arr = not_implemented "var"

let std arr = not_implemented "std"

let max arr = not_implemented "max"

let min arr = not_implemented "min"

let argmax arr = not_implemented "argmax"

let argmin arr = not_implemented "argmin"

let dsum arr dim = not_implemented "dsum"

let dmean arr dim = not_implemented "dmean"

let dvar arr dim = not_implemented "dvar"

let dstd arr dim = not_implemented "dstd"

let dmax arr dim = not_implemented "dmax"

let dmin arr dim = not_implemented "dmin"

let dargmax arr dim = not_implemented "dargmax"

let dargmin arr dim = not_implemented "dargmin"

let exp arr = not_implemented "exp"

let log arr = not_implemented "log"

let sqrt arr = not_implemented "sqrt"

let pow arr x = not_implemented "pow"

let expand_dims arr dim = not_implemented "expand_dims"

let squeeze arr = not_implemented "squeeze"

let map (arr: t) ~f :t= 
  {data= Array.map f arr.data; shape= arr.shape}

(* Reduction functions *)
let reduce_sum_to_shape (arr: t) (target_shape: int array) : t =
  let arr_shape = arr.shape in
  if Array.length arr_shape <> Array.length target_shape then
    failwith "Shapes must have the same number of dimensions for reduce_sum_to_shape";
  let axes_to_reduce = List.filter_map (fun (i, (dim_arr, dim_target)) ->
    if dim_arr <> dim_target then Some i else None
  ) (List.mapi (fun i dims -> (i, dims)) (Array.to_list (Array.combine arr_shape target_shape))) in
  List.fold_left (fun acc axis -> dsum acc axis) arr axes_to_reduce

  let negate arr =
    { data = Array.map (fun x -> -.x) arr.data; shape = arr.shape }
