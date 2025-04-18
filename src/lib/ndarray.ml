[@@@ocaml.warning "-27"]

(* let print_shape shape =
  Printf.printf "[%s]\n"
    (Stdlib.String.concat "; " (Stdlib.Array.to_list (Stdlib.Array.map string_of_int shape)))
;; *)

(* let print_data data =
  Printf.printf "[%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_float data)))
;; *)

type t = {
  data: float array;
  shape: int array;
}

(* Calculate total number of elements *)
let numel shape =
  Array.fold_left ( * ) 1 shape
;;

(* Calculate strides *)
let strides shape =
  let n = Array.length shape in
  let strides = Array.make n 1 in
  for i = n - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * shape.(i + 1)
  done;
  strides
;;

(* Custom function to compute the element-wise product of two arrays and sum them *)
let sum_multiplied a b =
  let len = Array.length a in
  if len <> Array.length b then
    failwith "Arrays must have the same length"
  else
    let acc = ref 0.0 in
    for i = 0 to len - 1 do
      acc := !acc +. (a.(i) *. b.(i))
    done;
    !acc
;;

(* Check if broadcasting is possible *)
let is_broadcastable shape1 shape2 =
  let len1 = Array.length shape1 in
  let len2 = Array.length shape2 in
  let max_len = Int.max len1 len2 in
  let s1 = Array.make max_len 1 in
  let s2 = Array.make max_len 1 in
  Array.blit shape1 0 s1 (max_len - len1) len1;
  Array.blit shape2 0 s2 (max_len - len2) len2;
  try
    for i = 0 to max_len - 1 do
      if not (s1.(i) = s2.(i) || s1.(i) = 1 || s2.(i) = 1) then
        raise Exit
    done;
    true
  with Exit -> false
;;

(* Calculate the shape after broadcasting *)
let broadcast_shape shape1 shape2 =
  let len1 = Array.length shape1 in
  let len2 = Array.length shape2 in
  let max_len = Int.max len1 len2 in
  let s1 = Array.make max_len 1 in
  let s2 = Array.make max_len 1 in
  Array.blit shape1 0 s1 (max_len - len1) len1;
  Array.blit shape2 0 s2 (max_len - len2) len2;
  Array.init max_len (fun i -> Int.max s1.(i) s2.(i))
;;

(* Broadcast addition *)
let add a b =
  if not (is_broadcastable a.shape b.shape) then
    failwith "Shapes are not broadcastable";
  let shape = broadcast_shape a.shape b.shape in
  let total_size = numel shape in
  let strides_res = strides shape in

  let len_a = Array.length a.shape in
  let len_b = Array.length b.shape in
  let len_res = Array.length shape in

  (* Fill shape1 and shape2 to match the broadcasted shape *)
  let s_a = Array.make len_res 1 in
  let s_b = Array.make len_res 1 in
  Array.blit a.shape 0 s_a (len_res - len_a) len_a;
  Array.blit b.shape 0 s_b (len_res - len_b) len_b;

  let strides_a = strides s_a in
  let strides_b = strides s_b in

  let result_data = Array.make total_size 0.0 in

  for idx = 0 to total_size - 1 do
    let indices = Array.make len_res 0 in
    let tmp = ref idx in
    for i = 0 to len_res - 1 do
      indices.(i) <- !tmp / strides_res.(i);
      tmp := !tmp mod strides_res.(i);
    done;

    let idx_a = ref 0 in
    let idx_b = ref 0 in
    for i = 0 to len_res - 1 do
      let idx_i = indices.(i) in
      let idx_ai = if s_a.(i) = 1 then 0 else idx_i in
      let idx_bi = if s_b.(i) = 1 then 0 else idx_i in
      idx_a := !idx_a + idx_ai * strides_a.(i);
      idx_b := !idx_b + idx_bi * strides_b.(i);
    done;

    result_data.(idx) <- a.data.(!idx_a) +. b.data.(!idx_b);
  done;

  { data = result_data; shape }
;;

(* Broadcast subtraction *)
let sub a b =
  if not (is_broadcastable a.shape b.shape) then
    failwith "Shapes are not broadcastable";
  let shape = broadcast_shape a.shape b.shape in
  let total_size = numel shape in
  let strides_res = strides shape in

  let len_a = Array.length a.shape in
  let len_b = Array.length b.shape in
  let len_res = Array.length shape in

  (* Fill shape1 and shape2 to match the broadcasted shape *)
  let s_a = Array.make len_res 1 in
  let s_b = Array.make len_res 1 in
  Array.blit a.shape 0 s_a (len_res - len_a) len_a;
  Array.blit b.shape 0 s_b (len_res - len_b) len_b;

  let strides_a = strides s_a in
  let strides_b = strides s_b in

  let result_data = Array.make total_size 0.0 in

  for idx = 0 to total_size - 1 do
    let indices = Array.make len_res 0 in
    let tmp = ref idx in
    for i = 0 to len_res - 1 do
      indices.(i) <- !tmp / strides_res.(i);
      tmp := !tmp mod strides_res.(i);
    done;

    let idx_a = ref 0 in
    let idx_b = ref 0 in
    for i = 0 to len_res - 1 do
      let idx_i = indices.(i) in
      let idx_ai = if s_a.(i) = 1 then 0 else idx_i in
      let idx_bi = if s_b.(i) = 1 then 0 else idx_i in
      idx_a := !idx_a + idx_ai * strides_a.(i);
      idx_b := !idx_b + idx_bi * strides_b.(i);
    done;

    result_data.(idx) <- a.data.(!idx_a) -. b.data.(!idx_b);
  done;

  { data = result_data; shape }
;;

(* Broadcasting multiplication *)
let mul a b =
  if not (is_broadcastable a.shape b.shape) then
    failwith "Shapes are not broadcastable";
  let shape = broadcast_shape a.shape b.shape in
  let total_size = numel shape in
  let strides_res = strides shape in

  let len_a = Array.length a.shape in
  let len_b = Array.length b.shape in
  let len_res = Array.length shape in

  (* Fill shape1 and shape2 to match the broadcasted shape *)
  let s_a = Array.make len_res 1 in
  let s_b = Array.make len_res 1 in
  Array.blit a.shape 0 s_a (len_res - len_a) len_a;
  Array.blit b.shape 0 s_b (len_res - len_b) len_b;

  let strides_a = strides s_a in
  let strides_b = strides s_b in

  let result_data = Array.make total_size 0.0 in

  for idx = 0 to total_size - 1 do
    let indices = Array.make len_res 0 in
    let tmp = ref idx in
    for i = 0 to len_res - 1 do
      indices.(i) <- !tmp / strides_res.(i);
      tmp := !tmp mod strides_res.(i);
    done;

    let idx_a = ref 0 in
    let idx_b = ref 0 in
    for i = 0 to len_res - 1 do
      let idx_i = indices.(i) in
      let idx_ai = if s_a.(i) = 1 then 0 else idx_i in
      let idx_bi = if s_b.(i) = 1 then 0 else idx_i in
      idx_a := !idx_a + idx_ai * strides_a.(i);
      idx_b := !idx_b + idx_bi * strides_b.(i);
    done;

    result_data.(idx) <- a.data.(!idx_a) *. b.data.(!idx_b);
  done;

  { data = result_data; shape }
;;

(* Broadcasting division *)
let div a b =
  if not (is_broadcastable a.shape b.shape) then
    failwith "Shapes are not broadcastable";
  let shape = broadcast_shape a.shape b.shape in
  let total_size = numel shape in
  let strides_res = strides shape in

  let len_a = Array.length a.shape in
  let len_b = Array.length b.shape in
  let len_res = Array.length shape in

  (* Fill shape1 and shape2 to match the broadcasted shape *)
  let s_a = Array.make len_res 1 in
  let s_b = Array.make len_res 1 in
  Array.blit a.shape 0 s_a (len_res - len_a) len_a;
  Array.blit b.shape 0 s_b (len_res - len_b) len_b;

  let strides_a = strides s_a in
  let strides_b = strides s_b in

  let result_data = Array.make total_size 0.0 in

  for idx = 0 to total_size - 1 do
    let indices = Array.make len_res 0 in
    let tmp = ref idx in
    for i = 0 to len_res - 1 do
      indices.(i) <- !tmp / strides_res.(i);
      tmp := !tmp mod strides_res.(i);
    done;

    let idx_a = ref 0 in
    let idx_b = ref 0 in
    for i = 0 to len_res - 1 do
      let idx_i = indices.(i) in
      let idx_ai = if s_a.(i) = 1 then 0 else idx_i in
      let idx_bi = if s_b.(i) = 1 then 0 else idx_i in
      idx_a := !idx_a + idx_ai * strides_a.(i);
      idx_b := !idx_b + idx_bi * strides_b.(i);
    done;

    result_data.(idx) <- a.data.(!idx_a) /. b.data.(!idx_b);
  done;

  { data = result_data; shape }
;;

(* Matrix multiplication with correct broadcasting and indexing *)
let matmul a b =
    let a_shape = a.shape in
    let b_shape = b.shape in
    let a_dim = Array.length a_shape in
    let b_dim = Array.length b_shape in
  
    (* Ensure input arrays have at least 1 dimension *)
    if a_dim < 1 || b_dim < 1 then
      failwith "matmul: Input arrays must have at least 1 dimension";
  
    (* Expand dimensions to at least 2D *)
    let a_shape_expanded =
      if a_dim == 1 then [|1; a_shape.(0)|] else a_shape in
    let b_shape_expanded =
      if b_dim == 1 then [|b_shape.(0); 1|] else b_shape in
  
    (* Update dimensions after expansion *)
    let a_dim = Array.length a_shape_expanded in
    let b_dim = Array.length b_shape_expanded in
  
    (* Get shapes and ensure inner dimensions match *)
    let a_shape_batch = Array.sub a_shape_expanded 0 (a_dim - 2) in
    let b_shape_batch = Array.sub b_shape_expanded 0 (b_dim - 2) in
    let a_m = a_shape_expanded.(a_dim - 2) in
    let a_k = a_shape_expanded.(a_dim - 1) in
    let b_k = b_shape_expanded.(b_dim - 2) in
    let b_n = b_shape_expanded.(b_dim - 1) in
  
    if a_k <> b_k then
      failwith (Printf.sprintf "matmul: Inner dimensions must match (got %d and %d)" a_k b_k);
  
    (* Compute broadcasted batch shape *)
    let batch_shape =
      if is_broadcastable a_shape_batch b_shape_batch then
        broadcast_shape a_shape_batch b_shape_batch
      else
        failwith "matmul: Batch dimensions are not broadcastable" in
  
    (* Compute output shape *)
    let output_shape = Array.concat [batch_shape; [|a_m; b_n|]] in
  
    (* Compute total number of elements *)
    let total_output_elements = numel output_shape in
    let result_data = Array.make total_output_elements 0.0 in
  
    (* Prepare strides *)
    let a_shape_broadcasted = Array.concat [batch_shape; [|a_m; a_k|]] in
    let b_shape_broadcasted = Array.concat [batch_shape; [|b_k; b_n|]] in
  
    let a_strides = strides a_shape_broadcasted in
    let b_strides = strides b_shape_broadcasted in
    let output_strides = strides output_shape in

    (* Function to compute index in original array, considering broadcasting *)
    let compute_index strides shape indices =
      let idx = ref 0 in
      for i = 0 to Array.length strides - 1 do
        let dim = shape.(i) in
        let idx_i = if dim = 1 then 0 else indices.(i) in
        idx := !idx + idx_i * strides.(i)
      done;
      !idx
    in
  
    (* Main computation loop *)
    let total_iterations = numel output_shape in
    for idx = 0 to total_iterations - 1 do
      (* Compute multi-dimensional indices for the output *)
      let indices = Array.make (Array.length output_shape) 0 in
      let tmp = ref idx in
      for i = 0 to Array.length output_shape - 1 do
        indices.(i) <- !tmp / output_strides.(i);
        tmp := !tmp mod output_strides.(i);
      done;
  
      (* Split indices into batch, m, and n indices *)
      let batch_indices = Array.sub indices 0 (Array.length batch_shape) in
      let i = indices.(Array.length indices - 2) in  (* m index *)
      let j = indices.(Array.length indices - 1) in  (* n index *)
  
      (* Compute indices for a and b *)
      let a_indices = Array.concat [batch_indices; [|i; 0|]] in
      let b_indices = Array.concat [batch_indices; [|0; j|]] in
  
      let sum = ref 0.0 in
      let a_size = numel a_shape in 
      let b_size = numel b_shape in 
      for k = 0 to a_k - 1 do
        a_indices.(Array.length a_indices - 1) <- k;
        b_indices.(Array.length b_indices - 2) <- k;
        let a_idx = compute_index a_strides a_shape_broadcasted a_indices mod a_size in
        let b_idx = compute_index b_strides b_shape_broadcasted b_indices mod b_size in
        let a_val = a.data.(a_idx) in
        let b_val = b.data.(b_idx) in
  
        sum := !sum +. a_val *. b_val;
      done;
  
      result_data.(idx) <- !sum;
    done;
  
    { data = result_data; shape = output_shape }
  ;;
  
(* Create an ndarray *)
let create (data: float array) (shape: int array) : t =
  let expected_size = Array.fold_left ( * ) 1 shape in
  if Array.length data <> expected_size then
    failwith "Data length does not match shape dimensions"
  else
    { data; shape }

let scaler (data: float) = 
  create [|data|] [||]

(* Generate a zero ndarray *)
let zeros (shape: int array) : t =
  let size = Array.fold_left ( * ) 1 shape in
  { data = Array.make size 0.0; shape }

(* Generate a ones ndarray *)
let ones (shape: int array) : t =
  let size = Array.fold_left ( * ) 1 shape in
  { data = Array.make size 1.0; shape }

let arange stop : t =
  let n = int_of_float stop in
  let data = Array.init n (fun i -> float_of_int i) in
  { data; shape = [|n|] }

let box_muller () =
  let u1 = Random.float 1.0 in
  let u2 = Random.float 1.0 in
  let r = sqrt (-2.0 *. log u1) in
  let theta = 2.0 *. Float.pi *. u2 in
  let z1 = r *. cos theta in
  z1

let rand (shape: int array) : t =
  let size = Array.fold_left ( * ) 1 shape in
  let data = Array.init size (fun _ -> (Random.float 2.0) -. 1.0) in
  { data; shape }

let randn (shape: int array) : t =
  let size = Array.fold_left ( * ) 1 shape in
  let data = Array.init size (fun _ -> box_muller ()) in
  { data; shape }

(* Xavier initialization *)
let xavier_init (shape: int array) : t =
  let size = Array.fold_left ( * ) 1 shape in
  let scale = sqrt (2.0 /. float_of_int size) in
  let data = Array.init size (fun _ -> (Random.float 1.0 -. 0.5) *. scale) in
  { data; shape }

(* Kaiming initialization *)
let kaiming_init (shape: int array) : t =
  let size = Array.fold_left ( * ) 1 shape in
  let scale = sqrt (2.0 /. float_of_int size) in
  let data = Array.init size (fun _ -> Random.float scale) in
  { data; shape }

(* Get the shape of the ndarray *)
let shape (arr: t) : int array =
  arr.shape

(* Get the dimensions of the ndarray *)
let dim (arr: t) : int =
  Array.length arr.shape

(* Get the value at a specified position in the ndarray *)
let at arr indices =
  let strides = strides arr.shape in
  let dims = Array.length arr.shape in
  let idx_len = Array.length indices in

  (* Check if the index length matches *)
  if idx_len <> dims then
    failwith (Printf.sprintf "Index length %d does not match number of dimensions %d" idx_len dims)
  else
    (* Check if each index is within the valid range *)
    let valid = ref true in
    for i = 0 to dims - 1 do
      if indices.(i) < 0 || indices.(i) >= arr.shape.(i) then
        valid := false
    done;
    if not !valid then
      failwith "Index out of bounds"
    else
      (* Calculate flat_index *)
      let flat_index =
        Array.fold_left (fun acc (i, stride) -> acc + i * stride) 0 (Array.combine indices strides)
      in
      (* Check if flat_index is within the data range *)
      if flat_index >= Array.length arr.data || flat_index < 0 then
        failwith "Flat index out of bounds"
      else
        arr.data.(flat_index)
;;

(* reshape function *)
let reshape (arr: t) (new_shape: int array) : t =
  let old_size = numel arr.shape in  (* Total number of elements in the original data *)

  (* Count the number of -1 in new_shape and calculate the product of known dimensions *)
  let num_neg_ones = ref 0 in
  let known_size = ref 1 in
  Array.iter (fun dim ->
    if dim = -1 then incr num_neg_ones
    else if dim > 0 then known_size := !known_size * dim
    else failwith "Reshape error: dimensions must be positive or -1"
  ) new_shape;

  (* Check if the number of -1s is valid *)
  if !num_neg_ones > 1 then
    failwith "Reshape error: only one dimension can be inferred (-1)";

  (* Calculate the dimension represented by -1 *)
  let inferred_shape =
    if !num_neg_ones = 1 then
      let inferred_dim = old_size / !known_size in
      if old_size mod !known_size <> 0 then
        failwith "Reshape error: total number of elements must remain unchanged";
      Array.map (fun dim -> if dim = -1 then inferred_dim else dim) new_shape
    else
      new_shape
  in

  (* Verify the total number of elements in the final shape *)
  let new_size = numel inferred_shape in
  if old_size <> new_size then
    failwith "Reshape error: total number of elements must remain unchanged";

  (* Return the new tensor structure *)
  { data = arr.data; shape = inferred_shape }
;;

(* Updates the element at a specified index in the ndarray *)
let set t idx value =
  let strides = strides t.shape in
  let dims = Array.length t.shape in
  let idx_len = Array.length idx in

  (* Check if the index length matches *)
  if idx_len <> dims then
    failwith (Printf.sprintf "Index length %d does not match number of dimensions %d" idx_len dims)
  else
    (* Check if each index is within the valid range *)
    let valid = ref true in
    for i = 0 to dims - 1 do
      if idx.(i) < 0 || idx.(i) >= t.shape.(i) then
        valid := false
    done;
    if not !valid then
      failwith "Index out of bounds"
    else
      (* Calculate flat_index *)
      let flat_index =
        Array.fold_left (fun acc (i, stride) -> acc + i * stride) 0 (Array.combine idx strides)
      in
      (* Check if flat_index is within the data range *)
      if flat_index >= Array.length t.data || flat_index < 0 then
        failwith "Flat index out of bounds"
      else
        t.data.(flat_index) <- value
;;

(* Returns the transpose of a 2D ndarray *)
let transpose t =
  if Array.length t.shape <> 2 then
    failwith "Transpose is only defined for 2D ndarrays"
  else
    let rows, cols = t.shape.(0), t.shape.(1) in
    let new_data = Array.make (rows * cols) 0.0 in
    for i = 0 to rows - 1 do
      for j = 0 to cols - 1 do
        new_data.(j * rows + i) <- t.data.(i * cols + j)
      done
    done;
    { data = new_data; shape = [|cols; rows|] }

(* Converts the ndarray to a flat float array *)
let to_array t =
  Array.copy t.data

let fill t value = 
  Array.fill t.data 0 (Array.length t.data) value 

let slice t slice_shape = 
  let len = Array.length t.shape in 
  if List.length slice_shape <> len then failwith "Slice: shape do not match"
  else 
    let get_shape tar_shape = 
      let shape_lst = List.fold_left (fun acc x -> acc @ [(snd x) - (fst x)]) [] tar_shape in 
      Array.of_list shape_lst
    in
    let final_shape = get_shape slice_shape in 
    if len = 1
    then 
      let (l, r) = List.nth slice_shape 0 in 
      let ans = ref [] in 
      for i = l to r - 1 do 
        ans := t.data.(i) :: !ans
      done;
      create (Array.of_list(List.rev !ans)) final_shape
    else if len = 2
    then
      let (l0, r0) = List.nth slice_shape 0 in 
      let (l1, r1) = List.nth slice_shape 1 in 
      let ans = ref [] in 
      for i0 = l0 to r0 - 1 do 
        for i1 = l1 to r1 - 1 do 
          ans := (at t [|i0; i1|]) :: !ans
        done;
      done;
      create (Array.of_list(List.rev !ans)) final_shape
    else if len = 3
      then
        let (l0, r0) = List.nth slice_shape 0 in 
        let (l1, r1) = List.nth slice_shape 1 in 
        let (l2, r2) = List.nth slice_shape 2 in 
        let ans = ref [] in 
        for i0 = l0 to r0 - 1 do 
          for i1 = l1 to r1 - 1 do 
            for i2 = l2 to r2 - 1 do 
              ans := (at t [|i0; i1; i2|]) :: !ans
            done;
          done;
        done;
        create (Array.of_list(List.rev !ans)) final_shape
    else if len = 4
      then
          let (l0, r0) = List.nth slice_shape 0 in 
          let (l1, r1) = List.nth slice_shape 1 in 
          let (l2, r2) = List.nth slice_shape 2 in 
          let (l3, r3) = List.nth slice_shape 3 in 
          let ans = ref [] in 
          for i0 = l0 to r0 - 1 do 
            for i1 = l1 to r1 - 1 do 
              for i2 = l2 to r2 - 1 do 
                for i3 = l3 to r3 - 1 do 
                  ans := (at t [|i0; i1; i2; i3|]) :: !ans
                done;
              done;
            done;
          done;
          create (Array.of_list(List.rev !ans)) final_shape
    else failwith "Slice: length longer than 4"


let sum t = 
  Array.fold_left (+.) 0.0 t.data

let mean t = 
  (sum t) /. (float_of_int (numel t.shape))

let var t = 
  let m = mean t in 
  let sum_val = Array.fold_left (fun acc x -> acc +. (x -. m) *. (x -. m)) 0.0 t.data in 
  sum_val /. (float_of_int (numel t.shape))

let std t = 
  sqrt (var t)

let max t = 
  let tmp = t.data.(0) in 
  let max_val = ref tmp in 
  for i = 0 to (numel t.shape) - 1 do 
    if !max_val < t.data.(i) then max_val := t.data.(i)
  done;
  !max_val

let min t = 
  let tmp = t.data.(0) in 
  let min_val = ref tmp in 
  for i = 0 to (numel t.shape) - 1 do 
    if !min_val > t.data.(i) then min_val := t.data.(i)
  done;
  !min_val

let argmax t = 
  let max_val = ref 0 in 
  for i = 0 to (numel t.shape) - 1 do 
    if t.data.(!max_val) < t.data.(i) then max_val := i
  done;
  !max_val

let argmin t = 
  let min_val = ref 0 in 
  for i = 0 to (numel t.shape) - 1 do 
    if t.data.(!min_val) > t.data.(i) then min_val := i
  done;
  !min_val

let dsum t dim =
  let shape = t.shape in
  let ndim = Array.length shape in
  if dim < 0 || dim >= ndim then failwith "Dimension out of range";
  let new_shape = Array.init (ndim - 1) (fun i -> if i < dim then shape.(i) else shape.(i + 1)) in
  let num_new_elements = Array.fold_left ( * ) 1 new_shape in
  let result_data = Array.make num_new_elements 0.0 in

  for idx = 0 to num_new_elements - 1 do
    (* Calculate the multi-dimensional index for the new shape *)
    let idx_new = Array.make (ndim - 1) 0 in
    let tmp_idx = ref idx in
    for i = (ndim - 2) downto 0 do
      let dim_size = new_shape.(i) in
      idx_new.(i) <- !tmp_idx mod dim_size;
      tmp_idx := !tmp_idx / dim_size;
    done;
    (* Map idx_new back to the original array's index *)
    let idx_full = Array.make ndim 0 in
    for i = 0 to ndim - 1 do
      if i < dim then
        idx_full.(i) <- idx_new.(i)
      else if i > dim then
        idx_full.(i) <- idx_new.(i - 1)
    done;
    (* Accumulate sum along the specified dimension *)
    let sum = ref 0.0 in
    for i_dim = 0 to shape.(dim) - 1 do
      idx_full.(dim) <- i_dim;
      let flat_idx = 
        let idx = ref 0 in
        for i = 0 to ndim - 1 do
          idx := !idx * shape.(i) + idx_full.(i)
        done;
        !idx
      in
      sum := !sum +. t.data.(flat_idx)
    done;
    result_data.(idx) <- !sum
  done;
  create result_data new_shape

let dmean t dim =
  let shape = t.shape in
  let ndim = Array.length shape in
  if dim < 0 || dim >= ndim then failwith "Dimension out of range";
  let new_shape = Array.init (ndim - 1) (fun i -> if i < dim then shape.(i) else shape.(i + 1)) in
  let num_new_elements = Array.fold_left ( * ) 1 new_shape in
  let result_data = Array.make num_new_elements 0.0 in
  let count = float_of_int shape.(dim) in

  for idx = 0 to num_new_elements - 1 do
    (* Calculate the multi-dimensional index for the new shape *)
    let idx_new = Array.make (ndim - 1) 0 in
    let tmp_idx = ref idx in
    for i = (ndim - 2) downto 0 do
      let dim_size = new_shape.(i) in
      idx_new.(i) <- !tmp_idx mod dim_size;
      tmp_idx := !tmp_idx / dim_size;
    done;
    (* Map back to the original array's index *)
    let idx_full = Array.make ndim 0 in
    for i = 0 to ndim - 1 do
      if i < dim then
        idx_full.(i) <- idx_new.(i)
      else if i > dim then
        idx_full.(i) <- idx_new.(i - 1)
    done;
    (* Calculate the mean *)
    let sum = ref 0.0 in
    for i_dim = 0 to shape.(dim) - 1 do
      idx_full.(dim) <- i_dim;
      let flat_idx =
        let idx_flat = ref 0 in
        for i = 0 to ndim - 1 do
          idx_flat := !idx_flat * shape.(i) + idx_full.(i)
        done;
        !idx_flat
      in
      sum := !sum +. t.data.(flat_idx)
    done;
    result_data.(idx) <- !sum /. count
  done;
  create result_data new_shape
  
let dvar t dim =
  let mean_result = dmean t dim in
  let shape = t.shape in
  let ndim = Array.length shape in
  let new_shape = mean_result.shape in
  let num_new_elements = Array.fold_left ( * ) 1 new_shape in
  let result_data = Array.make num_new_elements 0.0 in
  let count = float_of_int shape.(dim) in

  for idx = 0 to num_new_elements - 1 do
    (* Calculate the multi-dimensional index for the new shape *)
    let idx_new = Array.make (ndim - 1) 0 in
    let tmp_idx = ref idx in
    for i = (ndim - 2) downto 0 do
      let dim_size = new_shape.(i) in
      idx_new.(i) <- !tmp_idx mod dim_size;
      tmp_idx := !tmp_idx / dim_size;
    done;
    (* Map back to the original array's index *)
    let idx_full = Array.make ndim 0 in
    for i = 0 to ndim - 1 do
      if i < dim then
        idx_full.(i) <- idx_new.(i)
      else if i > dim then
        idx_full.(i) <- idx_new.(i - 1)
    done;
    (* Calculate variance *)
    let var_accum = ref 0.0 in
    for i_dim = 0 to shape.(dim) - 1 do
      idx_full.(dim) <- i_dim;
      let flat_idx =
        let idx_flat = ref 0 in
        for i = 0 to ndim - 1 do
          idx_flat := !idx_flat * shape.(i) + idx_full.(i)
        done;
        !idx_flat
      in
      let diff = t.data.(flat_idx) -. mean_result.data.(idx) in
      var_accum := !var_accum +. (diff ** 2.0)
    done;
    result_data.(idx) <- !var_accum /. count
  done;
  create result_data new_shape

let dstd t dim =
  let var_result = dvar t dim in
  { data = Array.map sqrt var_result.data; shape = var_result.shape }

let dmax t dim =
  let shape = t.shape in
  let ndim = Array.length shape in
  if dim < 0 || dim >= ndim then failwith "Dimension out of range";
  let new_shape = Array.init (ndim - 1) (fun i -> if i < dim then shape.(i) else shape.(i + 1)) in
  let num_new_elements = Array.fold_left ( * ) 1 new_shape in
  let result_data = Array.make num_new_elements Float.neg_infinity in

  for idx = 0 to num_new_elements - 1 do
    (* Calculate the multi-dimensional index for the new shape *)
    let idx_new = Array.make (ndim - 1) 0 in
    let tmp_idx = ref idx in
    for i = (ndim - 2) downto 0 do
      let dim_size = new_shape.(i) in
      idx_new.(i) <- !tmp_idx mod dim_size;
      tmp_idx := !tmp_idx / dim_size;
    done;
    (* Map back to the original array's index *)
    let idx_full = Array.make ndim 0 in
    for i = 0 to ndim - 1 do
      if i < dim then
        idx_full.(i) <- idx_new.(i)
      else if i > dim then
        idx_full.(i) <- idx_new.(i - 1)
    done;
    (* Calculate the maximum value *)
    let max_val = ref Float.neg_infinity in
    for i_dim = 0 to shape.(dim) - 1 do
      idx_full.(dim) <- i_dim;
      let flat_idx =
        let idx_flat = ref 0 in
        for i = 0 to ndim - 1 do
          idx_flat := !idx_flat * shape.(i) + idx_full.(i)
        done;
        !idx_flat
      in
      if t.data.(flat_idx) > !max_val then
        max_val := t.data.(flat_idx)
    done;
    result_data.(idx) <- !max_val
  done;
  create result_data new_shape  

let dargmax t dim =
  let shape = t.shape in
  let ndim = Array.length shape in
  if dim < 0 || dim >= ndim then failwith "Dimension out of range";
  let new_shape = Array.init (ndim - 1) (fun i -> if i < dim then shape.(i) else shape.(i + 1)) in
  let num_new_elements = Array.fold_left ( * ) 1 new_shape in
  let result_data = Array.make num_new_elements 0.0 in

  for idx = 0 to num_new_elements - 1 do
    (* Calculate the multi-dimensional index for the new shape *)
    let idx_new = Array.make (ndim - 1) 0 in
    let tmp_idx = ref idx in
    for i = (ndim - 2) downto 0 do
      let dim_size = new_shape.(i) in
      idx_new.(i) <- !tmp_idx mod dim_size;
      tmp_idx := !tmp_idx / dim_size;
    done;
    (* Map back to the original array's index *)
    let idx_full = Array.make ndim 0 in
    for i = 0 to ndim - 1 do
      if i < dim then
        idx_full.(i) <- idx_new.(i)
      else if i > dim then
        idx_full.(i) <- idx_new.(i - 1)
    done;
    (* Calculate the argmax index *)
    let max_val = ref Float.neg_infinity in
    let argmax_idx = ref 0 in
    for i_dim = 0 to shape.(dim) - 1 do
      idx_full.(dim) <- i_dim;
      let flat_idx =
        let idx_flat = ref 0 in
        for i = 0 to ndim - 1 do
          idx_flat := !idx_flat * shape.(i) + idx_full.(i)
        done;
        !idx_flat
      in
      if t.data.(flat_idx) > !max_val then begin
        max_val := t.data.(flat_idx);
        argmax_idx := i_dim;
      end
    done;
    result_data.(idx) <- float_of_int !argmax_idx
  done;
  create result_data new_shape

let dmin t dim =
  let shape = t.shape in
  let ndim = Array.length shape in
  if dim < 0 || dim >= ndim then failwith "Dimension out of range";
  let new_shape = Array.init (ndim - 1) (fun i -> if i < dim then shape.(i) else shape.(i + 1)) in
  let num_new_elements = Array.fold_left ( * ) 1 new_shape in
  let result_data = Array.make num_new_elements Float.infinity in

  for idx = 0 to num_new_elements - 1 do
    (* Calculate the multidimensional index in the new shape *)
    let idx_new = Array.make (ndim - 1) 0 in
    let tmp_idx = ref idx in
    for i = (ndim - 2) downto 0 do
      let dim_size = new_shape.(i) in
      idx_new.(i) <- !tmp_idx mod dim_size;
      tmp_idx := !tmp_idx / dim_size;
    done;
    (* Map back to the original array's index *)
    let idx_full = Array.make ndim 0 in
    for i = 0 to ndim - 1 do
      if i < dim then
        idx_full.(i) <- idx_new.(i)
      else if i > dim then
        idx_full.(i) <- idx_new.(i - 1)
    done;
    (* Calculate the minimum value *)
    let min_val = ref Float.infinity in
    for i_dim = 0 to shape.(dim) - 1 do
      idx_full.(dim) <- i_dim;
      let flat_idx =
        let idx_flat = ref 0 in
        for i = 0 to ndim - 1 do
          idx_flat := !idx_flat * shape.(i) + idx_full.(i)
        done;
        !idx_flat
      in
      if t.data.(flat_idx) < !min_val then
        min_val := t.data.(flat_idx)
    done;
    result_data.(idx) <- !min_val
  done;
  create result_data new_shape    
          
(* Element-wise exponential *)
let exp t =
  {
    data = Array.map exp t.data;
    shape = Array.copy t.shape;
  }

(* Element-wise natural logarithm *)
let log t =
  {
    data = Array.map log t.data;
    shape = Array.copy t.shape;
  }

(* Element-wise square root *)
let sqrt t =
  {
    data = Array.map sqrt t.data;
    shape = Array.copy t.shape;
  }

(* Element-wise power *)
let pow t x =
  {
    data = Array.map (fun y -> y ** x) t.data;
    shape = Array.copy t.shape;
  }

(* Adds a new dimension of size 1 at the specified dimension *)
let expand_dims t dim =
  let n = Array.length t.shape in
  if dim < 0 || dim > n then
    failwith "expand_dims: dimension out of range";
  let new_shape = Array.make (n + 1) 0 in
  Array.blit t.shape 0 new_shape 0 dim;
  new_shape.(dim) <- 1;
  Array.blit t.shape dim new_shape (dim + 1) (n - dim);
  {
    data = t.data;
    shape = new_shape;
  }

(** [expand_dims t dim] adds a new dimension of size 1 at the specified [dim] in [t]. *)

(* Removes dimensions of size 1 *)
let squeeze t =
  let new_shape = Array.of_list (List.filter (fun x -> x <> 1) (Array.to_list t.shape)) in
  if Array.length new_shape = 0 then
    { data = t.data; shape = [|1|] }  (* Ensure at least one dimension remains *)
  else
    { data = t.data; shape = new_shape }

(** [squeeze t] removes dimensions of size 1 from [t]. *)

let map (arr: t) ~f :t= 
  {data = Array.map f arr.data; shape= arr.shape}
let negate arr =
  {data = Array.map (fun x -> -.x) arr.data; shape = arr.shape }

(* Helper function to prepend ones to a shape array to match ranks *)
let pad_shape_to arr_shape target_shape =
  let arr_ndim = Array.length arr_shape in
  let tgt_ndim = Array.length target_shape in
  if arr_ndim = tgt_ndim then
    arr_shape, target_shape
  else if arr_ndim > tgt_ndim then
    (* Prepend ones to target_shape *)
    let padded = Array.init arr_ndim (fun i ->
      if i < (arr_ndim - tgt_ndim) then 1 else target_shape.(i - (arr_ndim - tgt_ndim))
    ) in
    arr_shape, padded
  else
    failwith "reduce_sum_to_shape: target shape cannot have more dimensions than arr"

let reduce_sum_to_shape (arr: t) (target_shape: int array) : t =
  let arr_shape = shape arr in
  (* If shapes differ in length, pad the shorter one with ones *)
  let arr_shape, tgt_shape = pad_shape_to arr_shape target_shape in

  if Array.length arr_shape <> Array.length tgt_shape then
    failwith "Shapes must match in number of dimensions after padding for reduce_sum_to_shape";

  (* Identify which axes need to be reduced *)
  let axes_to_reduce = 
    Array.to_list (Array.mapi (fun i (arr_dim, tgt_dim) ->
      if arr_dim <> tgt_dim then Some i else None
    ) (Array.combine arr_shape tgt_shape))
    |> List.filter_map (fun x -> x)
  in

  (* We should reduce from the largest axis to smallest axis to avoid index shifting issues *)
  let axes_to_reduce = List.rev (List.sort compare axes_to_reduce) in

  (* Perform the reductions *)
  List.fold_left (fun acc axis -> dsum acc axis) arr axes_to_reduce

let relu arr =
  let data_relu = Array.map (fun x -> if x > 0.0 then x else 0.0) arr.data in
  { data = data_relu; shape = arr.shape }

let to_string arr =
  let format_row row =
    Printf.sprintf "[%s]" (String.concat ", " (Array.to_list (Array.map string_of_float row)))
  in
  let rec format_ndarray data shape level =
    if Array.length shape = 0 then
      (* Empty shape means scalar value *)
      string_of_float data.(0)
    else if Array.length shape = 1 then
      (* Last dimension *)
      format_row (Array.init shape.(0) (fun i -> data.(i)))
    else
      (* Multidimensional array case *)
      let dim = shape.(0) in
      let sub_shape = Array.sub shape 1 (Array.length shape - 1) in
      let sub_size = Array.fold_left ( * ) 1 sub_shape in
      let sub_arrays = Array.init dim (fun i ->
        let start_idx = i * sub_size in
        Array.sub data start_idx sub_size
      ) in
      let indent = String.make (2 * level) ' ' in
      let sub_results = Array.map (fun sub -> format_ndarray sub sub_shape (level + 1)) sub_arrays in
      Printf.sprintf "[\n%s%s\n%s]" indent
        (String.concat (Printf.sprintf ",\n%s" indent) (Array.to_list sub_results))
        (if level > 0 then String.make (2 * (level - 1)) ' ' else "")
  in
  (* Check if shape and data size match *)
  if Array.fold_left ( * ) 1 arr.shape <> Array.length arr.data then
    failwith "Shape and data size mismatch"
  else
    format_ndarray arr.data arr.shape 0

let normalize t =
  let mean_vals = dmean t 0 in
  let std_vals = dstd t 0 in
  div (sub t mean_vals)  (add std_vals (scaler 0.0000001))

let transpose_last_two_dims t =
  let ndim = Array.length t.shape in
  if ndim < 2 then failwith "Tensor must have at least 2 dimensions to transpose the last two dimensions";
  let new_shape = Array.copy t.shape in
  new_shape.(ndim - 1) <- t.shape.(ndim - 2);
  new_shape.(ndim - 2) <- t.shape.(ndim - 1);
  let new_data = Array.make (Array.fold_left ( * ) 1 new_shape) 0.0 in

  let in_height = t.shape.(ndim - 2) in
  let in_width = t.shape.(ndim - 1) in
  let out_height = new_shape.(ndim - 2) in
  let out_width = new_shape.(ndim - 1) in

  for i = 0 to (Array.length t.data / (in_height * in_width)) - 1 do
    for h = 0 to in_height - 1 do
      for w = 0 to in_width - 1 do
        let new_h = w in
        let new_w = h in
        let old_idx = i * in_height * in_width + h * in_width + w in
        let new_idx = i * out_height * out_width + new_h * out_width + new_w in
        new_data.(new_idx) <- t.data.(old_idx)
      done
    done
  done;
  { data = new_data; shape = new_shape }

let conv2d input kernel ~stride ~padding =
  let batch_size, in_channels, in_height, in_width = input.shape.(0), input.shape.(1), input.shape.(2), input.shape.(3) in
  let out_channels, _, kernel_height, kernel_width = kernel.shape.(0), kernel.shape.(1), kernel.shape.(2), kernel.shape.(3) in

  let out_height = (in_height - kernel_height + 2 * padding) / stride + 1 in
  let out_width = (in_width - kernel_width + 2 * padding) / stride + 1 in

  let output = zeros [| batch_size; out_channels; out_height; out_width |] in

  for b = 0 to batch_size - 1 do
    for oc = 0 to out_channels - 1 do
      for oh = 0 to out_height - 1 do
        for ow = 0 to out_width - 1 do
          let h_start = oh * stride - padding in
          let w_start = ow * stride - padding in

          let sum = ref 0.0 in
          for ic = 0 to in_channels - 1 do
            for kh = 0 to kernel_height - 1 do
              for kw = 0 to kernel_width - 1 do
                let h = h_start + kh in
                let w = w_start + kw in
                if h >= 0 && h < in_height && w >= 0 && w < in_width then
                  sum := !sum +. input.data.((b * in_channels + ic) * in_height * in_width + h * in_width + w) *.
                                  kernel.data.((oc * in_channels + ic) * kernel_height * kernel_width + kh * kernel_width + kw)
              done
            done
          done;
          output.data.((b * out_channels + oc) * out_height * out_width + oh * out_width + ow) <- !sum
        done
      done
    done
  done;
  output

    
let flip_and_swap_kernel kernel =
  let out_channels = kernel.shape.(0) in
  let in_channels = kernel.shape.(1) in
  let kernel_height = kernel.shape.(2) in
  let kernel_width = kernel.shape.(3) in

  (* Create a new ndarray with the first two dimensions swapped *)
  let flipped_kernel = zeros [| in_channels; out_channels; kernel_height; kernel_width |] in

  (* Populate the new ndarray, rotating the last two dimensions *)
  for oc = 0 to out_channels - 1 do
    for ic = 0 to in_channels - 1 do
      for kh = 0 to kernel_height - 1 do
        for kw = 0 to kernel_width - 1 do
          (* Indices corresponding to a 180-degree rotation *)
          let flipped_kh = kernel_height - 1 - kh in
          let flipped_kw = kernel_width - 1 - kw in
          (* Swap the first two dimensions and adjust index calculations *)
          flipped_kernel.data.((ic * out_channels + oc) * kernel_height * kernel_width + flipped_kh * kernel_width + flipped_kw) <-
            kernel.data.((oc * in_channels + ic) * kernel_height * kernel_width + kh * kernel_width + kw)
        done
      done
    done
  done;
  flipped_kernel
  
let layerwise_convolution_with_doutput_as_kernel input doutput stride padding =
  let batch_size = input.shape.(0) in
  let out_channel = doutput.shape.(1) in
  let input_channel = input.shape.(1) in
  let input_h = input.shape.(2) in
  let input_w = input.shape.(3) in

  let output_h = doutput.shape.(2) in
  let output_w = doutput.shape.(3) in

  (* Calculate the height and width of the kernel *)
  let kernel_h = (input_h - output_h + 2 * padding) / stride + 1 in
  let kernel_w = (input_w - output_w + 2 * padding) / stride + 1 in

  (* Output tensor with shape (batch_size, out_channel, input_channel, kernel_h, kernel_w) *)
  let output = zeros [| batch_size; out_channel; input_channel; kernel_h; kernel_w |] in

  (* Perform the convolution operation *)
  for b = 0 to batch_size - 1 do
    for oc = 0 to out_channel - 1 do
      for ic = 0 to input_channel - 1 do
        for kh = 0 to kernel_h - 1 do
          for kw = 0 to kernel_w - 1 do
            (* Compute the starting position for the convolution operation *)
            let h_start = kh * stride - padding in
            let w_start = kw * stride - padding in

            (* Compute the convolution result at each position *)
            let sum = ref 0.0 in
            for dkh = 0 to output_h - 1 do
              for dkw = 0 to output_w - 1 do
                (* Check if the position is within the input range *)
                let h_pos = h_start + dkh in
                let w_pos = w_start + dkw in
                if h_pos >= 0 && h_pos < input_h && w_pos >= 0 && w_pos < input_w then
                  (* Perform the convolution operation *)
                  sum := !sum +. input.data.((((b) * input_channel + ic) * input_h + h_pos) * input_w + w_pos) *. 
                            doutput.data.((((b * out_channel + oc)) * output_h + dkh) * output_w + dkw)
              done
            done;

            (* Store the convolution result in the output tensor *)
            output.data.((((b * out_channel + oc) * input_channel + ic) * kernel_h + kh) * kernel_w + kw) <- !sum
          done
        done
      done
    done
  done;

  output

let image_scale t =
  let normalized_data = Array.copy t.data in

  for i = 0 to Array.length t.data - 1 do
    normalized_data.(i) <- ((t.data.(i) /. 255.0) -. 0.5) /. 0.5
  done;

  {data = normalized_data; shape = t.shape}

let l2_norm (nd: t) : float =
  Array.fold_left (fun acc x -> acc +. x *. x) 0.0 nd.data |> Stdlib.sqrt

let clip_by_norm (nd: t) (clip_value: float) : unit =
  let norm = l2_norm nd in
  if norm > clip_value then
    let scale = clip_value /. norm in
    Array.iteri (fun i x -> nd.data.(i) <- x *. scale) nd.data

let clip_by_range (nd: t) (min_value: float) (max_value: float) : unit =
  Array.iteri (fun i x ->
    if Float.abs x > max_value then
      nd.data.(i) <- (if x > 0. then max_value else -.max_value)
    else if Float.abs x < min_value then
      nd.data.(i) <- (if x > 0. then min_value else -.min_value)
  ) nd.data