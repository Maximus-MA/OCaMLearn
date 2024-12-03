(* src/backprop.ml *)

[@@@ocaml.warning "-27"]
open Core

type tensor = Tensor.t

(* Placeholder for unimplemented functions *)
let not_implemented feature_name =
  failwith (feature_name ^ " is not yet implemented")


let backprop ts =
  Tensor.set_grad ts (Ndarray.ones [|1|]);
  let rec topo u vis res =
    if List.mem vis u ~equal:phys_equal 
      then vis, res
    else 
      let vis = u::vis in
      let vis, res = List.fold Tensor.(u.prev)
        ~init:(vis, res) 
        ~f:(fun acc v -> 
          let vis, res = acc in
          topo v vis res) in 
      (vis, u::res) in
  let _, res = topo ts [] [] in 
  List.iter res 
    ~f:(fun t -> 
      match t.backward_fn with 
      | None -> () 
      | Some f -> f ();)

      
        
      
