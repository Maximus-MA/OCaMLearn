[@@@ocaml.warning "-27"]
(* let print_shape shape =
  Printf.printf "[%s]\n"
    (Stdlib.String.concat "; " (Stdlib.Array.to_list (Stdlib.Array.map string_of_int shape)))
;; *)

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

let scaler x =
  {
    data = Ndarray.scaler x;
    grad = Ndarray.scaler 0.0;
    requires_grad = false;
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
    t.grad <- Ndarray.add t.grad grad

let zero_grad t =
  t.grad <- Ndarray.zeros @@ shape t

let get_grad t =
  t.grad

let set_grad t grad =
  t.grad <- grad

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
        accumulate_grad t1 grad_t1;
      if t2.requires_grad then
        let grad_t2 = Ndarray.reduce_sum_to_shape grad_output (Ndarray.shape t2.data) in
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
        accumulate_grad t1 grad_t1;
      if t2.requires_grad then
        let grad_t2 = (Ndarray.negate grad_output |> Ndarray.reduce_sum_to_shape) (shape t2) in
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
        accumulate_grad t1 grad_t1;
      if t2.requires_grad then
        let grad_t2 = (Ndarray.mul grad_output t1.data |> Ndarray.reduce_sum_to_shape) (Ndarray.shape t2.data) in
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
        accumulate_grad t1 grad_t1;
      if t2.requires_grad then
        let numerator = Ndarray.mul grad_output t1.data in
        let denominator = Ndarray.mul t2.data t2.data in
        let grad_t2 = (Ndarray.div numerator denominator |> Ndarray.negate |> Ndarray.reduce_sum_to_shape) (Ndarray.shape t2.data) in
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
        accumulate_grad t1 grad_t1;
      if t2.requires_grad then
        let grad_t2 = (Ndarray.matmul (Ndarray.transpose t1.data) grad_output
                      |> Ndarray.reduce_sum_to_shape) (Ndarray.shape t2.data) in
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
      accumulate_grad t grad_t;
    );
  res

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
      let grad_input = match dim with
        | Some d -> Ndarray.expand_dims grad_output d
        | None -> not_implemented "sum"
      in
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
  if requires_grad then
    t_new.backward_fn <- Some (fun () ->
      let grad_output = t_new.grad in
      let num_elements = match dim with
        | Some d -> float_of_int (Ndarray.shape t.data).(d)
        | None -> float_of_int (Ndarray.numel (Ndarray.shape t.data))
      in
      let grad_input = match dim with
        | Some d -> Ndarray.expand_dims grad_output d
        | None -> not_implemented "mean" 
      in
      let grad_single = (Ndarray.scaler num_elements) in
      let grad_input = Ndarray.div grad_input grad_single in
      accumulate_grad t grad_input;
    );
  t_new
  
let dsum t dim =
  sum ~dim:dim t

let dmax t dim =
  let data = Ndarray.dmax t.data dim in
  let requires_grad = t.requires_grad in
  let prev = [t] in
  let res = create ~data ~requires_grad ~prev in
  if requires_grad then
    res.backward_fn <- Some (fun () ->
      let grad_output = res.grad in
      let grad_input = Ndarray.expand_dims grad_output dim in
      accumulate_grad t grad_input);
  res

(* Element-wise mathematical functions *)
let exp t =
  let data = Ndarray.exp t.data in
  let requires_grad = t.requires_grad in
  let t_new = {
    data;
    grad = Ndarray.zeros (Ndarray.shape data);
    requires_grad;
    backward_fn = None;
    prev = [t];
  } in
  if requires_grad then
    t_new.backward_fn <- Some (fun () ->
      let grad_output = t_new.grad in
      let grad_input = Ndarray.mul grad_output t_new.data in
      accumulate_grad t grad_input);
  t_new



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
  if requires_grad then
    t_new.backward_fn <- Some (fun () ->
      let grad_output = t_new.grad in
      let grad_input = Ndarray.negate grad_output in
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
    let mask = Ndarray.map ~f:(fun x -> if (x > 0.0) then 1.0 else 0.0) t.data in
    let grad_input = Ndarray.mul grad_output mask in
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
      let grad_input = Ndarray.sub grad_output (Ndarray.mul softmax_probs (Ndarray.expand_dims sum_grad (Ndarray.dim sum_grad))) in
      accumulate_grad t grad_input;
    );
  t_new

let slice t ranges  =
  let data = Ndarray.slice t.data ranges in
  let requires_grad = t.requires_grad in
  let prev = [] in
  let res = create ~data ~requires_grad ~prev in
  (* res.backward_fn <- Some (fun () ->
    let grad_output = res.grad in
    let grad_input = Ndarray.slice grad_output ranges in
    accumulate_grad t grad_input); *)
  res.backward_fn <- None;
  res

let conv2d t kernel ~stride ~padding =
  let data = Ndarray.conv2d t.data kernel.data ~stride ~padding in
  let requires_grad = t.requires_grad || kernel.requires_grad in
  let prev = [t; kernel] in
  let res = create ~data ~requires_grad ~prev in
  res.backward_fn <- Some (fun () ->
    let grad_output = res.grad in
    if t.requires_grad then
      let grad_t = Ndarray.conv2d grad_output (Ndarray.flip_and_swap_kernel kernel.data) ~stride:stride ~padding:padding in
      accumulate_grad t grad_t;
    if kernel.requires_grad then
      let layerwise_convolution_result = Ndarray.layerwise_convolution_with_doutput_as_kernel t.data grad_output stride padding in
      let grad_kernel = Ndarray.dsum layerwise_convolution_result 0 in
      accumulate_grad kernel grad_kernel
  );
  res

let meanpool2d t ~kernel_size ~stride =
  let input_shape = t.data.shape in
  let batch_size = input_shape.(0) in
  let channels = input_shape.(1) in
  let input_height = input_shape.(2) in
  let input_width = input_shape.(3) in

  (* Calculate the output height and width *)
  let output_height = (input_height - kernel_size) / stride + 1 in
  let output_width = (input_width - kernel_size) / stride + 1 in

  (* Create output data *)
  let output_data = Ndarray.zeros [| batch_size; channels; output_height; output_width |] in

  (* Forward pass: Iterate over pooling windows and compute the average *)
  for b = 0 to batch_size - 1 do
    for c = 0 to channels - 1 do
      for h = 0 to output_height - 1 do
        for w = 0 to output_width - 1 do
          (* Starting indices for the current pooling window *)
          let h_start = h * stride in
          let w_start = w * stride in
          let h_end = Core.min (h_start + kernel_size) input_height in
          let w_end = Core.min (w_start + kernel_size) input_width in

          (* Extract the window and compute the average *)
          let sum = ref 0.0 in
          for i = h_start to h_end - 1 do
            for j = w_start to w_end - 1 do
              let idx = (b * channels * input_height * input_width) +
                        (c * input_height * input_width) +
                        (i * input_width) + j in
              sum := !sum +. t.data.data.(idx);
            done
          done;

          let num_elements = float_of_int ((h_end - h_start) * (w_end - w_start)) in
          let mean_value = !sum /. num_elements in

          (* Store the average value in the output tensor *)
          let output_idx = (b * channels * output_height * output_width) +
                            (c * output_height * output_width) +
                            (h * output_width) + w in
          output_data.data.(output_idx) <- mean_value;
        done
      done
    done
  done;

  (* Create the output Tensor *)
  let res = {
    data = { data = output_data.data; shape = [| batch_size; channels; output_height; output_width |] };
    requires_grad = t.requires_grad;
    grad = Ndarray.zeros [| batch_size; channels; output_height; output_width |];
    backward_fn = None;
    prev = [t];
  } in

  (* If backward propagation is needed, define the gradient computation logic *)
  if t.requires_grad then
    res.backward_fn <- Some (fun () ->
      let grad_output = res.grad in
      let grad_input = Ndarray.zeros t.data.shape in

      (* Backward propagation: Distribute the gradient evenly to each element in the window *)
      for b = 0 to batch_size - 1 do
        for c = 0 to channels - 1 do
          for h = 0 to output_height - 1 do
            for w = 0 to output_width - 1 do
              (* Starting indices for the current window *)
              let h_start = h * stride in
              let w_start = w * stride in
              let h_end = Core.min (h_start + kernel_size) input_height in
              let w_end = Core.min (w_start + kernel_size) input_width in

              let grad_value = grad_output.data.((b * channels * output_height * output_width) +
                                                  (c * output_height * output_width) +
                                                  (h * output_width) + w) in
              let num_elements = float_of_int ((h_end - h_start) * (w_end - w_start)) in
              let distributed_grad = grad_value /. num_elements in

              (* Distribute the gradient back to the input tensor *)
              for i = h_start to h_end - 1 do
                for j = w_start to w_end - 1 do
                  let idx = (b * channels * input_height * input_width) +
                            (c * input_height * input_width) +
                            (i * input_width) + j in
                  grad_input.data.(idx) <- grad_input.data.(idx) +. distributed_grad;
                done
              done;
            done
          done
        done
      done;

      (* Accumulate gradients *)
      accumulate_grad t grad_input
    );
  res
let to_string t =
  Printf.sprintf "Tensor {data = %s, requires_grad = %b}" (Ndarray.to_string t.data) t.requires_grad

(* let expand t new_shape =
  not_implemented "expand"

let concatenate ts dim =
  not_implemented "concatenate"

let split t dim =
  not_implemented "split"

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

let argmin t =
  not_implemented "argmin"

let dmean t dim =
  not_implemented "dmean"

let dvariance t dim =
  not_implemented "dvariance"

let dstd t dim =
  not_implemented "dstd"

let dnormalize t dim =
  not_implemented "dnormalize"

let log t =
  not_implemented "log"

let pow t x =
  not_implemented "pow"

let sqrt t =
  not_implemented "sqrt"

(* Utility functions *)
let clone t =
  not_implemented "clone"

let can_broadcast t1 t2 =
  not_implemented "can_broadcast"

let detach t =
  not_implemented "detach"

let dmin t dim =
  not_implemented "dmin"

let dargmax t dim =
  not_implemented "dargmax"

let dargmin t dim =
  not_implemented "dargmin" *)