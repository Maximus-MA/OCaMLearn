(* src/tensor.ml *)
[@@@ocaml.warning "-27"]

open Core

type ndarray = Ndarray.t

type t = {
  mutable data: ndarray;
  mutable grad: ndarray;
  requires_grad: bool;
  mutable backward_fn: (unit -> unit) option;
  prev: t list;
}

(* Placeholder for unimplemented functions *)
let not_implemented feature_name =
  failwith (feature_name ^ " is not yet implemented")

(* Creation functions *)
let create ~data ~requires_grad ~prev =
  {
    data;
    grad = (Ndarray.zeros (Ndarray.shape data));
    requires_grad;
    backward_fn = None;  (* Will set later *)
    prev;
  }

let from_ndarray ?(requires_grad=true) data  =
  {
    data;
    grad = (Ndarray.zeros (Ndarray.shape data));
    requires_grad = requires_grad;
    backward_fn = None;
    prev = [];
  }
  

let zeros shape =
  let data = Ndarray.zeros shape in
  {
    data;
    grad = (Ndarray.zeros shape);
    requires_grad = true;
    backward_fn = None;
    prev = [];
  }
let ones shape =
  let data = Ndarray.ones shape in
  {
    data;
    grad = (Ndarray.zeros shape);
    requires_grad = true;
    backward_fn = None;
    prev = [];
  }

let arange stop =
  let data = Ndarray.arange stop in
  {
    data;
    grad = Ndarray.zeros (Ndarray.shape data);
    requires_grad = true;
    backward_fn = None;
    prev = [];
  }

let rand shape =
  let data = Ndarray.rand shape in
  {
    data;
    grad = (Ndarray.zeros shape);
    requires_grad = true;
    backward_fn = None;
    prev = [];
  }

let xavier_init shape =
  let data = Ndarray.xavier_init shape in
  {
    data;
    grad = (Ndarray.zeros shape);
    requires_grad = true;
    backward_fn = None;
    prev = [];
  }

let he_init shape =
  let data = Ndarray.kaiming_init shape in  (* Assuming kaiming_init is equivalent to he_init *)
  {
    data;
    grad = (Ndarray.zeros shape);
    requires_grad = true;
    backward_fn = None;
    prev = [];
  }
  

(* Tensor information and properties *)
let ndim tensor =
  Ndarray.dim tensor.data

let get_data t =
  t.data

let shape t =
  Ndarray.shape t.data

let requires_grad t =
  t.requires_grad

let get t idx =
  Ndarray.at t.data idx 

let set t idx value =
  Ndarray.set t.data idx value

(* Gradient-related functions *)
let accumulate_grad t grad =
    (* Printf.printf "[%s]\n" (Ndarray.to_string grad); *)
    (Ndarray.print_shape t.grad.shape);
    t.grad <- Ndarray.add t.grad grad
let zero_grad t =
  t.grad <- Ndarray.zeros @@ shape t

let get_grad t =
  t.grad

let set_grad t grad =
  t.grad <- grad

(* let clip_grad t max_val =
  let clipped_grad = Ndarray.map ~f:(fun x -> max (-.max_val) (min x max_val)) !t.grad in 
  t.grad := ref(clipped_grad) *)

let clip_grad t max_val =
  not_implemented"clip"

(* Element-wise operations *)
let add t1 t2 =
  let data = Ndarray.add t1.data t2.data in
  let requires_grad = t1.requires_grad || t2.requires_grad in
  let t = {
    data;
    grad = (Ndarray.zeros (Ndarray.shape data));
    requires_grad;
    backward_fn = None;  (* Will set later *)
    prev = [t1; t2];
  } in
  if requires_grad then
    t.backward_fn <- Some (fun () ->
      let grad_output = t.grad in
      if t1.requires_grad then
        let grad_t1 = Ndarray.reduce_sum_to_shape grad_output (Ndarray.shape t1.data) in
        Printf.printf "[%s]\n" (Ndarray.to_string grad_t1);
        accumulate_grad t1 grad_t1;
      if t2.requires_grad then
        let grad_t2 = Ndarray.reduce_sum_to_shape grad_output (Ndarray.shape t2.data) in
        Printf.printf "[%s]\n" (Ndarray.to_string grad_t2);
        accumulate_grad t2 grad_t2;
		);
  t


let sub t1 t2 =
  let data = Ndarray.sub t1.data t2.data in
  let requires_grad = t1.requires_grad || t2.requires_grad in
  let t = {
    data;
    grad =(Ndarray.zeros (Ndarray.shape data));
    requires_grad;
    backward_fn = None;
    prev = [t1; t2];
  } in
  if requires_grad then
    t.backward_fn <- Some (fun () ->
      let grad_output = t.grad in
      if t1.requires_grad then
        let grad_t1 = Ndarray.reduce_sum_to_shape grad_output (Ndarray.shape t1.data) in
        Printf.printf "[%s]\n" (Ndarray.to_string grad_t1);
        accumulate_grad t1 grad_t1;
      if t2.requires_grad then
        let grad_t2 = (Ndarray.negate grad_output |> Ndarray.reduce_sum_to_shape) (shape t2) in
        Printf.printf "[%s]\n" (Ndarray.to_string grad_t2);
        accumulate_grad t2 grad_t2;
    );
  t

let mul t1 t2 =
  let data = Ndarray.mul t1.data t2.data in
  let requires_grad = t1.requires_grad || t2.requires_grad in
  let t = {
    data;
    grad = Ndarray.zeros (Ndarray.shape data);
    requires_grad;
    backward_fn = None;
    prev = [t1; t2];
  } in
  if requires_grad then
    t.backward_fn <- Some (fun () ->
      let grad_output = t.grad in
      if t1.requires_grad then
        let grad_t1 = (Ndarray.mul grad_output t2.data |> Ndarray.reduce_sum_to_shape) (Ndarray.shape t1.data) in
        Printf.printf "[%s]\n" (Ndarray.to_string grad_t1);
        accumulate_grad t1 grad_t1;
      if t2.requires_grad then
        let grad_t2 = (Ndarray.mul grad_output t1.data |> Ndarray.reduce_sum_to_shape) (Ndarray.shape t2.data) in
        Printf.printf "[%s]\n" (Ndarray.to_string grad_t2);
        accumulate_grad t2 grad_t2;
    );
  t
  

  let div t1 t2 =
    let data = Ndarray.div t1.data t2.data in
    let requires_grad = t1.requires_grad || t2.requires_grad in
    let t = {
      data;
      grad = Ndarray.zeros (Ndarray.shape data);
      requires_grad;
      backward_fn = None;
      prev = [t1; t2];
    } in
    if requires_grad then
      t.backward_fn <- Some (fun () ->
        let grad_output = t.grad in
        if t1.requires_grad then
          let grad_t1 = (Ndarray.div grad_output t2.data |> Ndarray.reduce_sum_to_shape) (Ndarray.shape t1.data) in
          Printf.printf "[%s]\n" (Ndarray.to_string grad_t1);
          accumulate_grad t1 grad_t1;
        if t2.requires_grad then
          let numerator = Ndarray.mul grad_output t1.data in
          let denominator = Ndarray.mul t2.data t2.data in
          let grad_t2 = (Ndarray.div numerator denominator |> Ndarray.negate |> Ndarray.reduce_sum_to_shape) (Ndarray.shape t2.data) in
          Printf.printf "[%s]\n" (Ndarray.to_string grad_t2);
          accumulate_grad t2 grad_t2;
      );
    t
  

(* Matrix operations *)
let matmul t1 t2 =
  let data = Ndarray.matmul t1.data t2.data in
  let requires_grad = t1.requires_grad || t2.requires_grad in
  let t = {
    data;
    grad = (Ndarray.zeros (Ndarray.shape data));
    requires_grad;
    backward_fn = None;
    prev = [t1; t2];
  } in
  if requires_grad then
    t.backward_fn <- Some (fun () ->
      let grad_output = t.grad in
      if t1.requires_grad then
        let grad_t1 = (Ndarray.matmul grad_output (Ndarray.transpose t2.data)
                      |> Ndarray.reduce_sum_to_shape) (Ndarray.shape t1.data) in
        Printf.printf "[%s]\n" (Ndarray.to_string grad_t1); 
        accumulate_grad t1 grad_t1;
      if t2.requires_grad then
        let grad_t2 = (Ndarray.matmul (Ndarray.transpose t1.data) grad_output
                      |> Ndarray.reduce_sum_to_shape) (Ndarray.shape t2.data) in
        Printf.printf "[%s]\n" (Ndarray.to_string grad_t2);
        accumulate_grad t2 grad_t2;
    );
  t

  let transpose t =
    let data = Ndarray.transpose t.data in
    let requires_grad = t.requires_grad in
    let t_new = {
      data;
      grad = (Ndarray.zeros (Ndarray.shape data));
      requires_grad;
      backward_fn = None;
      prev = [t];
    } in
    if requires_grad then
      t_new.backward_fn <- Some (fun () ->
        let grad_output = t_new.grad in
        let grad_t = Ndarray.transpose grad_output in
        Printf.printf "[%s]\n" (Ndarray.to_string grad_t);
        accumulate_grad t grad_t;
      );
    t_new

(* Tensor shape operations *)
let reshape t ~shape =
  let data = Ndarray.reshape t.data shape in
  let requires_grad = t.requires_grad in
  let prev = [t] in
  let res = create ~data ~requires_grad ~prev in
  if requires_grad then
    res.backward_fn <- Some (fun () ->
      let grad_output = res.grad in
      let grad_t = Ndarray.reshape grad_output (Ndarray.shape t.data) in
      Printf.printf "[%s]\n" (Ndarray.to_string grad_t);
      accumulate_grad t grad_t;
    );
  res

let expand t new_shape =
  not_implemented "expand"

let concatenate ts dim =
  not_implemented "concatenate"

let split t dim =
  not_implemented "split"

(* Mathematical functions *)
let sum ?dim t =
  let data = match dim with
    | Some d -> Ndarray.dsum t.data d
    | None -> not_implemented "sum"
  in
  let requires_grad = t.requires_grad in
  let t_new = {
    data;
    grad = Ndarray.zeros data.shape;
    requires_grad;
    backward_fn = None;
    prev = [t];
  } in
  if requires_grad then
    t_new.backward_fn <- Some (fun () ->
      let grad_output = t_new.grad in
      (Ndarray.print_shape grad_output.shape);
      (Ndarray.print_shape t.grad.shape);
      let grad_input = match dim with
        | Some d -> Ndarray.expand_dims grad_output d
        | None -> not_implemented "sum"
      in
      Printf.printf "[%s]\n" (Ndarray.to_string grad_input);
      accumulate_grad t grad_input;
    );  t_new

  let mean ?dim t =
    let data = match dim with
      | Some d -> Ndarray.dmean t.data d
      | None -> not_implemented "mean"
    in
    let requires_grad = t.requires_grad in
    let t_new = {
      data;
      grad = Ndarray.zeros data.shape;
      requires_grad;
      backward_fn = None;
      prev = [t];
    } in
    Ndarray.print_shape t_new.grad.shape;
    if requires_grad then
      t_new.backward_fn <- Some (fun () ->
        let grad_output = t_new.grad in
        (Ndarray.print_shape grad_output.shape);
        let num_elements = match dim with
          | Some d -> float_of_int (Ndarray.shape t.data).(d)
          | None -> float_of_int (Ndarray.numel (Ndarray.shape t.data))
        in
        let grad_input = match dim with
          | Some d -> Ndarray.expand_dims grad_output d
          | None -> not_implemented "mean" 
        in
        let grad_single = (Ndarray.create_float num_elements) in
        let grad_input = Ndarray.div grad_input grad_single in
        Printf.printf "[%s]\n" (Ndarray.to_string grad_input);
        accumulate_grad t grad_input;
      );
    t_new
  

let variance t =
  not_implemented "variance"

let std t =
  not_implemented "std"

let normalize t =
  not_implemented "normalize"

let max t =
  not_implemented "max"

let min t =
  not_implemented "min"

let argmax t =
  not_implemented "argmax"
  (* TODO *)

let argmin t =
  not_implemented "argmin"

let dsum t dim =
  sum ~dim:dim t

let dmean t dim =
  not_implemented "dmean"

let dvariance t dim =
  not_implemented "dvariance"

let dstd t dim =
  not_implemented "dstd"

let dnormalize t dim =
  not_implemented "dnormalize"

let dmax t dim =
  not_implemented "dmax"

let dmin t dim =
  not_implemented "dmin"

let dargmax t dim =
  not_implemented "dargmax"

let dargmin t dim =
  not_implemented "dargmin"

(* Element-wise mathematical functions *)
let exp t =
  not_implemented "exp"
  (* TODO *)

let log t =
  not_implemented "log"
  (* TODO *)

let pow t x =
  not_implemented "pow"
  (* TODO *)

let sqrt t =
  not_implemented "sqrt"

(* Utility functions *)
let clone t =
  not_implemented "clone"

let can_broadcast t1 t2 =
  not_implemented "can_broadcast"

let detach t =
  not_implemented "detach"

let neg t =
  let data = Ndarray.negate t.data in
  let requires_grad = t.requires_grad in
  let t_new = {
    data;
    grad = Ndarray.zeros (Ndarray.shape data);
    requires_grad;
    backward_fn = None;
    prev = [t];
  } in
  Ndarray.print_shape t_new.grad.shape;
  if requires_grad then
    t_new.backward_fn <- Some (fun () ->
      let grad_output = t_new.grad in
      (Ndarray.print_shape grad_output.shape);
      let grad_input = Ndarray.negate grad_output in
      Printf.printf "[%s]\n" (Ndarray.to_string grad_input);
      accumulate_grad t grad_input;
    );
  t_new
  

let relu t =
  let data = Ndarray.relu t.data in
  let requires_grad = t.requires_grad in
  let prev = [t] in
  let res = create ~data ~requires_grad ~prev in
  res.backward_fn <- Some (fun () ->
    let grad_output = res.grad in
    let mask = Ndarray.map ~f:(fun x -> if Float.(x > 0.0) then 1.0 else 0.0) t.data in
    let grad_input = Ndarray.mul grad_output mask in
    Printf.printf "[%s]\n" (Ndarray.to_string grad_input);
    accumulate_grad t grad_input);
  res

  let softmax t =
    let max_vals = Ndarray.dmax t.data (Ndarray.dim t.data - 1) in
    let shifted_logits = Ndarray.sub t.data  max_vals in
    let exp_logits = Ndarray.exp shifted_logits in
    let sum_exp = Ndarray.dsum exp_logits (Ndarray.dim t.data - 1) in
    let probs = Ndarray.div exp_logits sum_exp in
  
    let requires_grad = t.requires_grad in
    let t_new = {
      data = probs;
      grad = Ndarray.zeros (Ndarray.shape probs);
      requires_grad;
      backward_fn = None;
      prev = [t];
    } in
  
    if requires_grad then
      t_new.backward_fn <- Some (fun () ->
        let grad_output = t_new.grad in
        (* 
           dSoftmax = Softmax * (grad_output - sum(grad_output * Softmax))
        *)
        let sum_grad = Ndarray.dsum (Ndarray.mul grad_output t_new.data) (Ndarray.dim t.data - 1) in
        let grad_input = Ndarray.mul t_new.data (Ndarray.sub grad_output (Ndarray.expand_dims sum_grad (Ndarray.dim sum_grad))) in
        accumulate_grad t grad_input;
      );
    t_new
  
  let log_softmax t =
    let max_vals = Ndarray.dmax t.data (Ndarray.dim t.data - 1) in
    let shifted_logits = Ndarray.sub t.data (Ndarray.expand_dims max_vals (Ndarray.dim max_vals))in
    let sum_exp = Ndarray.dsum (Ndarray.exp shifted_logits) (Ndarray.dim t.data - 1) in
    let log_probs = Ndarray.sub shifted_logits (Ndarray.log (Ndarray.expand_dims sum_exp (Ndarray.dim sum_exp))) in

    let requires_grad = t.requires_grad in
    let t_new = {
      data = log_probs;
      grad = Ndarray.zeros (Ndarray.shape log_probs);
      requires_grad;
      backward_fn = None;
      prev = [t];
    } in

    if requires_grad then
      t_new.backward_fn <- Some (fun () ->
        let grad_output = t_new.grad in
        (* Gradient of log_softmax is grad_output - exp(log_probs) * sum(grad_output) *)
        let softmax_probs = Ndarray.exp t_new.data in
        let sum_grad = Ndarray.dsum (Ndarray.mul grad_output softmax_probs) (Ndarray.dim t.data - 1) in
        Ndarray.print_shape sum_grad.shape;
        let grad_input = Ndarray.sub grad_output (Ndarray.mul softmax_probs (Ndarray.expand_dims sum_grad (Ndarray.dim sum_grad))) in
        accumulate_grad t grad_input;
      );
    t_new

let slice t ranges  =
  let data = Ndarray.slice t.data ranges in
  let requires_grad = t.requires_grad in
  let prev = [t] in
  let res = create ~data ~requires_grad ~prev in
  res.backward_fn <- Some (fun () ->
    let grad_output = res.grad in
    let grad_input = Ndarray.slice grad_output ranges in
    accumulate_grad t grad_input);
  res

(* 
  type t = {
  mutable data: ndarray;                    (* The tensor's data as an ndarray *)
  mutable grad: ndarray;             (* Gradient of the tensor, if required *)
  requires_grad: bool;              (* Indicates if the tensor requires gradient computation *)
  mutable backward_fn: (unit -> unit) option;  (* Function to compute gradients for backpropagation *)
  prev: t list;                     (* List of previous tensors for backpropagation *)
} *)

let to_string t =
  Printf.sprintf "Tensor {data = %s, requires_grad = %b}" (Ndarray.to_string t.data) t.requires_grad