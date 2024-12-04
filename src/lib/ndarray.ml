(* src/ndarray.ml *)
[@@@ocaml.warning "-27"]

let print_shape shape =
  Printf.printf "[%s]\n"
    (Stdlib.String.concat "; " (Stdlib.Array.to_list (Stdlib.Array.map string_of_int shape)))
;;

type t = {
  data: float array;
  shape: int array;
}

(* 计算元素总数 *)
let numel shape =
  Array.fold_left ( * ) 1 shape
;;

(* 计算步长 *)
let strides shape =
  let n = Array.length shape in
  let strides = Array.make n 1 in
  for i = n - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * shape.(i + 1)
  done;
  strides
;;

(* 自定义的两个数组的元素乘积并求和函数 *)
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

(* 检查是否可广播 *)
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

(* 计算广播后的形状 *)
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

(* 广播加法 *)
let add a b =
  if not (is_broadcastable a.shape b.shape) then
    failwith "Shapes are not broadcastable";
  let shape = broadcast_shape a.shape b.shape in
  let total_size = numel shape in
  let strides_res = strides shape in

  let len_a = Array.length a.shape in
  let len_b = Array.length b.shape in
  let len_res = Array.length shape in

  (* 填充 shape1 和 shape2 以匹配广播后的 shape *)
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

(* 广播减法 *)
let sub a b =
  print_shape a.shape;
  print_shape b.shape;
  if not (is_broadcastable a.shape b.shape) then
    failwith "Shapes are not broadcastable";
  let shape = broadcast_shape a.shape b.shape in
  let total_size = numel shape in
  let strides_res = strides shape in

  let len_a = Array.length a.shape in
  let len_b = Array.length b.shape in
  let len_res = Array.length shape in

  (* 填充 shape1 和 shape2 以匹配广播后的 shape *)
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

(* 广播乘法 *)
let mul a b =
  if not (is_broadcastable a.shape b.shape) then
    failwith "Shapes are not broadcastable";
  let shape = broadcast_shape a.shape b.shape in
  let total_size = numel shape in
  let strides_res = strides shape in

  let len_a = Array.length a.shape in
  let len_b = Array.length b.shape in
  let len_res = Array.length shape in

  (* 填充 shape1 和 shape2 以匹配广播后的 shape *)
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

(* 广播除法 *)
let div a b =
  if not (is_broadcastable a.shape b.shape) then
    failwith "Shapes are not broadcastable";
  let shape = broadcast_shape a.shape b.shape in
  let total_size = numel shape in
  let strides_res = strides shape in

  let len_a = Array.length a.shape in
  let len_b = Array.length b.shape in
  let len_res = Array.length shape in

  (* 填充 shape1 和 shape2 以匹配广播后的 shape *)
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
    print_shape a_shape;
    print_shape b_shape;
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
        (* print_int_array a_indices; *)
        (* print_int_array b_indices; *)
        let a_idx = compute_index a_strides a_shape_broadcasted a_indices mod a_size in
        let b_idx = compute_index b_strides b_shape_broadcasted b_indices mod b_size in
        (* print_int_array a_strides; *)
        (* print_int_array a_shape_broadcasted; *)
        (* Printf.printf "idx_a = %d, idx_b = %d\n" a_idx b_idx; *)
        let a_val = a.data.(a_idx) in
        let b_val = b.data.(b_idx) in
  
        sum := !sum +. a_val *. b_val;
      done;
  
      result_data.(idx) <- !sum;
    done;
  
    { data = result_data; shape = output_shape }
  ;;
  
(* 创建一个ndarray *)
let create (data: float array) (shape: int array) : t =
  let expected_size = Array.fold_left ( * ) 1 shape in
  if Array.length data <> expected_size then
    failwith "Data length does not match shape dimensions"
  else
    { data; shape }

let create_float (data: float) = 
  create [|data|] [|1|]

let create_int (data: int) = 
  create_float (float_of_int data)

(* 生成全零的ndarray *)
let zeros (shape: int array) : t =
  let size = Array.fold_left ( * ) 1 shape in
  { data = Array.make size 0.0; shape }

(* 生成全一的ndarray *)
let ones (shape: int array) : t =
  let size = Array.fold_left ( * ) 1 shape in
  { data = Array.make size 1.0; shape }

(* 生成随机值的ndarray *)
let rand (shape: int array) : t =
  let size = Array.fold_left ( * ) 1 shape in
  let data = Array.init size (fun _ -> Random.float 1.0) in
  { data; shape }

(* Xavier 初始化 *)
let xavier_init (shape: int array) : t =
  let size = Array.fold_left ( * ) 1 shape in
  let scale = sqrt (2.0 /. float_of_int size) in
  let data = Array.init size (fun _ -> (Random.float 1.0 -. 0.5) *. scale) in
  { data; shape }

(* Kaiming 初始化 *)
let kaiming_init (shape: int array) : t =
  let size = Array.fold_left ( * ) 1 shape in
  let scale = sqrt (2.0 /. float_of_int size) in
  let data = Array.init size (fun _ -> Random.float scale) in
  { data; shape }

(* 获取ndarray的形状 *)
let shape (arr: t) : int array =
  arr.shape

(* 获取ndarray的维数 *)
let dim (arr: t) : int =
  Array.length arr.shape

(* 获取ndarray指定位置的值 *)
let at (arr: t) (indices: int array) : float =
  let strides = Array.make (Array.length arr.shape) 1 in
  for i = Array.length arr.shape - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * arr.shape.(i + 1)
  done;
  let index = ref 0 in
  for i = 0 to Array.length indices - 1 do
    index := !index + indices.(i) * strides.(i)
  done;
  arr.data.(!index)
;;

(* reshape 函数 *)
let reshape (arr: t) (new_shape: int array) : t =
  let old_size = numel arr.shape in
  let new_size = numel new_shape in
  if old_size <> new_size then
    failwith "Reshape error: total number of elements must remain unchanged"
  else
    { data = arr.data; shape = new_shape }
;;


(* ------------------------------------------------------------------------------------------- *)

(* Updates the element at a specified index in the ndarray *)
let set t idx value =
  (* Calculate the flat index from multidimensional indices *)
  let flat_index = 
    Array.fold_left (fun (acc, stride) i -> (acc + i * stride, stride * t.shape.(i))) (0, 1) idx
    |> fst 
  in
  t.data.(flat_index) <- value

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
    (* 计算新形状下的多维索引 *)
    let idx_new = Array.make (ndim - 1) 0 in
    let tmp_idx = ref idx in
    for i = (ndim - 2) downto 0 do
      let dim_size = new_shape.(i) in
      idx_new.(i) <- !tmp_idx mod dim_size;
      tmp_idx := !tmp_idx / dim_size;
    done;
    (* 将 idx_new 映射回原始数组的索引 *)
    let idx_full = Array.make ndim 0 in
    for i = 0 to ndim - 1 do
      if i < dim then
        idx_full.(i) <- idx_new.(i)
      else if i > dim then
        idx_full.(i) <- idx_new.(i - 1)
    done;
    (* 在指定维度上累加 *)
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
    (* 计算新形状下的多维索引 *)
    let idx_new = Array.make (ndim - 1) 0 in
    let tmp_idx = ref idx in
    for i = (ndim - 2) downto 0 do
      let dim_size = new_shape.(i) in
      idx_new.(i) <- !tmp_idx mod dim_size;
      tmp_idx := !tmp_idx / dim_size;
    done;
    (* 映射回原始数组的索引 *)
    let idx_full = Array.make ndim 0 in
    for i = 0 to ndim - 1 do
      if i < dim then
        idx_full.(i) <- idx_new.(i)
      else if i > dim then
        idx_full.(i) <- idx_new.(i - 1)
    done;
    (* 计算均值 *)
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
    (* 计算新形状下的多维索引 *)
    let idx_new = Array.make (ndim - 1) 0 in
    let tmp_idx = ref idx in
    for i = (ndim - 2) downto 0 do
      let dim_size = new_shape.(i) in
      idx_new.(i) <- !tmp_idx mod dim_size;
      tmp_idx := !tmp_idx / dim_size;
    done;
    (* 映射回原始数组的索引 *)
    let idx_full = Array.make ndim 0 in
    for i = 0 to ndim - 1 do
      if i < dim then
        idx_full.(i) <- idx_new.(i)
      else if i > dim then
        idx_full.(i) <- idx_new.(i - 1)
    done;
    (* 计算方差 *)
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
      (* 计算新形状下的多维索引 *)
      let idx_new = Array.make (ndim - 1) 0 in
      let tmp_idx = ref idx in
      for i = (ndim - 2) downto 0 do
        let dim_size = new_shape.(i) in
        idx_new.(i) <- !tmp_idx mod dim_size;
        tmp_idx := !tmp_idx / dim_size;
      done;
      (* 映射回原始数组的索引 *)
      let idx_full = Array.make ndim 0 in
      for i = 0 to ndim - 1 do
        if i < dim then
          idx_full.(i) <- idx_new.(i)
        else if i > dim then
          idx_full.(i) <- idx_new.(i - 1)
      done;
      (* 计算最大值 *)
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
    
let dmin t dim =
  let shape = t.shape in
  let ndim = Array.length shape in
  if dim < 0 || dim >= ndim then failwith "Dimension out of range";
  let new_shape = Array.init (ndim - 1) (fun i -> if i < dim then shape.(i) else shape.(i + 1)) in
  let num_new_elements = Array.fold_left ( * ) 1 new_shape in
  let result_data = Array.make num_new_elements Float.infinity in

  for idx = 0 to num_new_elements - 1 do
    (* 计算新形状下的多维索引 *)
    let idx_new = Array.make (ndim - 1) 0 in
    let tmp_idx = ref idx in
    for i = (ndim - 2) downto 0 do
      let dim_size = new_shape.(i) in
      idx_new.(i) <- !tmp_idx mod dim_size;
      tmp_idx := !tmp_idx / dim_size;
    done;
    (* 映射回原始数组的索引 *)
    let idx_full = Array.make ndim 0 in
    for i = 0 to ndim - 1 do
      if i < dim then
        idx_full.(i) <- idx_new.(i)
      else if i > dim then
        idx_full.(i) <- idx_new.(i - 1)
    done;
    (* 计算最小值 *)
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

(** [exp t] computes the exponential of each element in [t]. *)

(* Element-wise natural logarithm *)
let log t =
  {
    data = Array.map log t.data;
    shape = Array.copy t.shape;
  }

(** [log t] computes the natural logarithm of each element in [t]. *)

(* Element-wise square root *)
let sqrt t =
  {
    data = Array.map sqrt t.data;
    shape = Array.copy t.shape;
  }

(** [sqrt t] computes the square root of each element in [t]. *)

(* Element-wise power *)
let pow t x =
  {
    data = Array.map (fun y -> y ** x) t.data;
    shape = Array.copy t.shape;
  }

(** [pow t x] raises each element in [t] to the power of [x]. *)

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


(* #TODO *)
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



(* #TODO *)
let relu arr =
  let data_relu = Array.map (fun x -> if x > 0.0 then x else 0.0) arr.data in
  { data = data_relu; shape = arr.shape }


let to_string arr =
  let rec format_ndarray data shape level =
    if Array.length shape = 1 then
      let row = Array.init shape.(0) (fun i -> data.(i)) in
      Printf.sprintf "[%s]" (String.concat ", " (Array.to_list (Array.map string_of_float row)))
    else
      let dim = shape.(0) in
      let sub_shape = Array.sub shape 1 (Array.length shape - 1) in
      let sub_size = Array.fold_left ( * ) 1 sub_shape in
      let sub_arrays = Array.init dim (fun i ->
        let start_idx = i * sub_size in
        let _ = start_idx + sub_size in
        Array.sub data start_idx sub_size
      ) in
      let indent = String.make (2 * level) ' ' in
      let sub_results = Array.map (fun sub -> format_ndarray sub sub_shape (level + 1)) sub_arrays in
      Printf.sprintf "[\n%s%s\n%s]" indent
        (String.concat (Printf.sprintf ",\n%s" indent) (Array.to_list sub_results))
        (if level > 0 then String.make (2 * (level - 1)) ' ' else "") in
  format_ndarray arr.data arr.shape 0
  
