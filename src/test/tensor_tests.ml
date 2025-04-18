open OUnit2
open Core

module T = Tensor
module N = Ndarray

let float_array_equal ~(epsilon:float) arr1 arr2 =
  let len1 = Array.length arr1 in
  let len2 = Array.length arr2 in
  if len1 <> len2 then false else
    Array.for_alli arr1 ~f:(fun i x ->
      Float.(abs (x -. arr2.(i)) < epsilon))

let float_ndarray_equal ~epsilon nd1 nd2 =
  let arr1 = N.to_array nd1 in
  let arr2 = N.to_array nd2 in
  float_array_equal ~epsilon arr1 arr2

let shape_equal s1 s2 =
  let len1 = Array.length s1 in
  let len2 = Array.length s2 in
  len1 = len2 && Array.for_alli s1 ~f:(fun i x -> x = s2.(i))
  
  
let eps = 1e-7

(* Basic tests are assumed to be defined in a similar manner as before. *)
(* Below we add tests for broadcasting scenarios. *)

let test_broadcast_add _ =
  (* Shape [2;3] + [3] should broadcast to [2;3].
     For example: [ [1,2,3],
                    [4,5,6] ]
     plus [10,20,30]
     = [ [11,22,33],
         [14,25,36] ]
  *)
  let t1 = T.from_ndarray (N.create [|1.;2.;3.;4.;5.;6.|] [|2;3|]) in
  let t2 = T.from_ndarray (N.create [|10.;20.;30.|] [|3|]) in
  let out = T.add t1 t2 in
  let expected = N.create [|11.;22.;33.;14.;25.;36.|] [|2;3|] in
  assert_bool "Broadcast add forward"
    (float_ndarray_equal ~epsilon:eps out.T.data expected);

  (* Backward *)
  T.zero_grad t1; T.zero_grad t2; T.zero_grad out;
  out.T.grad <- N.ones [|2;3|]; (* gradient is all ones *)
  (match out.T.backward_fn with
   | Some fn -> fn ()
   | None -> assert_failure "No backward for add");

  (* Gradient checks:
     For t1 (same shape), gradient is just the out gradient: all ones in shape [2;3].
     For t2 (shape [3]), the gradient must be reduced (summing over the first dimension):
     sum over the 2 rows: each element in t2’s grad = sum of two gradients from each row
     So t2 grad = [2;2;2]
  *)
  let expected_grad_t1 = N.ones [|2;3|] in
  let expected_grad_t2 = N.create [|2.;2.;2.|] [|3|] in
  assert_bool "Grad t1 broadcast add" (float_ndarray_equal ~epsilon:eps t1.T.grad expected_grad_t1);
  assert_bool "Grad t2 broadcast add" (float_ndarray_equal ~epsilon:eps t2.T.grad expected_grad_t2)


let test_broadcast_mul _ =
  (* Shape [2;3] * [3] should broadcast:
     t1 = [[1,2,3],[4,5,6]]
     t2 = [10,20,30]

     out = [[1*10,2*20,3*30],
            [4*10,5*20,6*30]]
         = [[10,40,90],
            [40,100,180]]
  *)
  let t1 = T.from_ndarray (N.create [|1.;2.;3.;4.;5.;6.|] [|2;3|]) in
  let t2 = T.from_ndarray (N.create [|10.;20.;30.|] [|3|]) in
  let out = T.mul t1 t2 in
  let expected = N.create [|10.;40.;90.;40.;100.;180.|] [|2;3|] in
  assert_bool "Broadcast mul forward"
    (float_ndarray_equal ~epsilon:eps out.T.data expected);

  (* Backward *)
  T.zero_grad t1; T.zero_grad t2; T.zero_grad out;
  out.T.grad <- N.ones [|2;3|]; 
  (match out.T.backward_fn with
   | Some fn -> fn ()
   | None -> assert_failure "No backward for mul");

  (* Grad checks:
     dOut/dt1 = t2 broadcasted = [[10,20,30],[10,20,30]]
     So t1.grad = sum over out gradient * t2:
        all ones * t2 = [[10,20,30],[10,20,30]]
     t1.grad shape [2;3], equals [[10,20,30],[10,20,30]]

     dOut/dt2 = t1 broadcasted, but must be reduced to shape [3]:
        sum over dimension 0:
        Column 0: 1 + 4 = 5
        Column 1: 2 + 5 = 7
        Column 2: 3 + 6 = 9
     t2.grad = [5,7,9]
  *)
  let expected_grad_t1 = N.create [|10.;20.;30.;10.;20.;30.|] [|2;3|] in
  let expected_grad_t2 = N.create [|5.;7.;9.|] [|3|] in

  assert_bool "Grad t1 broadcast mul"
    (float_ndarray_equal ~epsilon:eps t1.T.grad expected_grad_t1);
  assert_bool "Grad t2 broadcast mul"
    (float_ndarray_equal ~epsilon:eps t2.T.grad expected_grad_t2)


let test_broadcast_sub _ =
  (* Shape [2;3] - [3]:
     t1 = [[10,11,12],[13,14,15]]
     t2 = [1,2,3]
     out = [[9,9,9],[12,12,12]]

  *)
  let t1 = T.from_ndarray (N.create [|10.;11.;12.;13.;14.;15.|] [|2;3|]) in
  let t2 = T.from_ndarray (N.create [|1.;2.;3.|] [|3|]) in
  let out = T.sub t1 t2 in
  let expected = N.create [|9.;9.;9.;12.;12.;12.|] [|2;3|] in
  assert_bool "Broadcast sub forward"
    (float_ndarray_equal ~epsilon:eps out.T.data expected);

  (* Backward *)
  T.zero_grad t1; T.zero_grad t2; T.zero_grad out;
  out.T.grad <- N.ones [|2;3|]; 
  (match out.T.backward_fn with
   | Some fn -> fn ()
   | None -> assert_failure "No backward for sub");

  (* Grad checks:
     For t1: Grad is all ones, same shape [2;3].
     For t2: Grad is negative ones summed over dim 0:
       Each element in t2.grad = sum of -1 from each row:
       t2.grad = [-2, -2, -2]
  *)
  let expected_grad_t1 = N.ones [|2;3|] in
  let expected_grad_t2 = N.create [|-2.;-2.;-2.|] [|3|] in
  assert_bool "Grad t1 broadcast sub"
    (float_ndarray_equal ~epsilon:eps t1.T.grad expected_grad_t1);
  assert_bool "Grad t2 broadcast sub"
    (float_ndarray_equal ~epsilon:eps t2.T.grad expected_grad_t2)


(* If division also supports broadcasting, test similarly *)
let test_broadcast_div _ =
  (* Shape [2;3] / [3]:
     t1 = [[2,4,6],[8,10,12]]
     t2 = [1,2,3]
     out = [[2/1,4/2,6/3],[8/1,10/2,12/3]]
         = [[2,2,2],[8,5,4]]
  *)
  let t1 = T.from_ndarray (N.create [|2.;4.;6.;8.;10.;12.|] [|2;3|]) in
  let t2 = T.from_ndarray (N.create [|1.;2.;3.|] [|3|]) in
  let out = T.div t1 t2 in
  let expected = N.create [|2.;2.;2.;8.;5.;4.|] [|2;3|] in
  assert_bool "Broadcast div forward"
    (float_ndarray_equal ~epsilon:eps out.T.data expected);

  (* Backward *)
  T.zero_grad t1; T.zero_grad t2; T.zero_grad out;
  out.T.grad <- N.ones [|2;3|]; 
  (match out.T.backward_fn with
   | Some fn -> fn () 
   | None -> assert_failure "No backward for div");

  (* Grad checks:
     dOut/dt1 = 1/t2 broadcasted:
       = [[1/1,1/2,1/3],[1/1,1/2,1/3]]
       = [[1,0.5,0.3333],[1,0.5,0.3333]]

     sum over entire t1 since same shape, no reduction needed.

     dOut/dt2 = - (t1 / t2^2):
       For each element:
       row 0: [-2/1^2, -4/2^2, -6/3^2] = [-2, -1, -6/9 = -0.6667]
       row 1: [-8/1, -10/4, -12/9] = [-8, -2.5, -1.3333]
       
     Sum over the 2 rows for t2’s grad:
       t2[0]: -2 + (-8) = -10
       t2[1]: -1 + (-2.5) = -3.5
       t2[2]: -0.6667 + (-1.3333) = -2.0 (approx)
  *)
  let expected_grad_t1 = N.create [|1.;0.5;1./.3.;1.;0.5;1./.3.|] [|2;3|] in
  let expected_grad_t2 = N.create [|-10.;-3.5; -2.|] [|3|] in
  assert_bool "Grad t1 broadcast div"
    (float_ndarray_equal ~epsilon:1e-5 t1.T.grad expected_grad_t1);
  assert_bool "Grad t2 broadcast div"
    (float_ndarray_equal ~epsilon:1e-5 t2.T.grad expected_grad_t2)

    
  (* Test creation functions *)
  let test_create_basic _ =
    let data = N.ones [|2;2|] in
    let t = T.create ~data ~requires_grad:true ~prev:[] in
    assert_bool "check data shape" (shape_equal (N.shape t.T.data) [|2;2|]);
    assert_bool "check grad shape" (shape_equal (N.shape t.T.grad) [|2;2|]);
    assert_equal true (T.requires_grad t)
  
  let test_from_ndarray_requires_grad_false _ =
    let data = N.arange 6. in
    let t = T.from_ndarray ~requires_grad:false data in
    assert_equal false (T.requires_grad t)
  
  let test_zeros_shape _ =
    let t = T.zeros [|3;4|] in
    assert_bool "zeros shape" (shape_equal (N.shape t.T.data) [|3;4|]);
    assert_bool "zeros values" (float_ndarray_equal ~epsilon:eps t.T.data (N.zeros [|3;4|]))
  
  let test_ones_shape _ =
    let t = T.ones [|2;3|] in
    assert_bool "ones shape" (shape_equal (N.shape t.T.data) [|2;3|]);
    assert_bool "ones values" (float_ndarray_equal ~epsilon:eps t.T.data (N.ones [|2;3|]))
  
  let test_rand_shape _ =
    let shape = [|2;2|] in
    let t = T.rand shape in
    assert_bool "rand shape" (shape_equal (N.shape t.T.data) shape);
    (* Can't predict values, but can check range if Ndarray.rand is uniform in [0,1) *)
    let arr = N.to_array t.T.data in
    assert_bool "all values in [0,1)" Float.(Array.for_all arr ~f:(fun x -> x >= 0.0 && x < 1.0))
  
  let test_xavier_init_shape _ =
    let shape = [|3;3|] in
    let t = T.xavier_init shape in
    assert_bool "xavier shape" (shape_equal (N.shape t.T.data) shape)
  
  let test_he_init_shape _ =
    let shape = [|4;5|] in
    let t = T.he_init shape in
    assert_bool "he_init shape" (shape_equal (N.shape t.T.data) shape)
  
  (* Test get and set *)
  let test_get_set_values _ =
    let t = T.ones [|2;2|] in
    T.set t [|0;1|] 5.0;
    assert_equal 5.0 (T.get t [|0;1|]);
    T.set t [|1;0|] (-3.0);
    assert_equal (-3.0) (T.get t [|1;0|])
  
  (* Test arithmetic operations *)
  let test_add_grad_accumulate _ =
    let t1 = T.ones [|2;2|] in
    let t1 = T.transpose t1 in 
    let t2 = T.ones [|2;2|] in
    let out = T.add t1 t2 in
    out.T.grad <- N.ones [|2;2|]; 
    (match out.T.backward_fn with Some fn -> fn () | None -> ());
    (* Check gradient accumulation by calling backward again *)
    out.T.grad <- N.ones [|2;2|];
    (match out.T.backward_fn with Some fn -> fn () | None -> ());
    (* Now t1 and t2 grads should be 2 * ones *)
    let expected =N.zeros [|2;2|] in
    N.fill expected 2.0;
    assert_bool "grad accumulation t1" (float_ndarray_equal ~epsilon:eps t1.T.grad expected);
    assert_bool "grad accumulation t2" (float_ndarray_equal ~epsilon:eps t2.T.grad expected)
  
  (* let test_sub_nonrequiring_grad _ =
    let t1 = T.from_ndarray ~requires_grad:false (N.ones [|2;2|]) in
    let t2 = T.ones [|2;2|] in
    let out = T.sub t1 t2 in
    out.T.grad <- N.ones [|2;2|];
    (match out.T.backward_fn with Some fn -> fn () | None -> ());
    (* t1 does not require grad, so it stays zero *)
    let expected_t1_grad = N.zeros [|2;2|] in
    let expected_t2_grad = N.full [|2;2|] (-1.0) in
    assert_bool "t1 no grad" (float_ndarray_equal ~epsilon:eps t1.T.grad expected_t1_grad);
    assert_bool "t2 grad" (float_ndarray_equal ~epsilon:eps t2.T.grad expected_t2_grad)
   *)
  
  (* Test matmul with higher dimensions if supported, or just standard *)
  let test_matmul_non_square _ =
    let t1 = T.from_ndarray (N.create [|1.;2.;3.;4.;5.;6.|] [|2;3|]) in
    let t2 = T.from_ndarray (N.create [|7.;8.;9.;10.;11.;12.|] [|3;2|]) in
    (* out shape = [2;2] *)
    let out = T.matmul t1 t2 in
    (* Manual matmul:
        [ [1*7+2*9+3*11, 1*8+2*10+3*12],
          [4*7+5*9+6*11, 4*8+5*10+6*12] ]
        Wait carefully:
        It's (2x3) * (3x2):
          out[0,0] = 1*7 + 2*9 + 3*11 = 7 +18 +33 =58
          out[0,1] = 1*8 + 2*10 + 3*12 =8+20+36=64
          out[1,0] = 4*7 +5*9 +6*11 =28+45+66=139
          out[1,1] = 4*8 +5*10+6*12=32+50+72=154
      *)
    let expected = N.create [|58.;64.;139.;154.|] [|2;2|] in
    assert_bool "matmul forward" (float_ndarray_equal ~epsilon:eps out.T.data expected);
    out.T.grad <- N.ones [|2;2|];
    (match out.T.backward_fn with Some fn -> fn () | None -> ());
    (* Check gradients roughly:
        d(t1) = dOut * t2^T
        t2^T = [[7,9,11],[8,10,12]]
        dOut = ones:
          d(t1) shape = [2;3]:
          row0: sum along row of t2^T: (7+8)=15, (9+10)=19, (11+12)=23
          Actually, we must do full matmul:
          d(t1) = [[1,1];[1,1]] * t2^T = 
            For first row of t1:
              [ (1*7+1*8), (1*9+1*10), (1*11+1*12) ] = [15,19,23]
            For second row of t1:
              the same since gradient is all ones: [15,19,23]
  
        so t1.grad = [[15,19,23],[15,19,23]]
  
        d(t2) = t1^T * dOut
        t1^T = [[1,4],[2,5],[3,6]]
        dOut = ones:
          Each element of t2:
            [ (1+4),(2+5),(3+6) ; same for second column ]
          Actually:
          d(t2) shape [3;2]:
            For column0: sum first column of t1^T with out grads:
            t2 grad col0: 
              [ (1*1+4*1) , (2*1+5*1), (3*1+6*1)] = [5,7,9]
          For column1: same:
              [5,7,9]
  
        t2.grad = [[5,5],[7,7],[9,9]] after proper dimension checks. Wait carefully:
        d(t2) = (t1^T (2x3)) * (dOut (2x2)):
          t1^T is (3x2), dOut is (2x2), so result (3x2)
          d(t2)[0,0] = (1*1 +4*1)=5; d(t2)[0,1]=(1*1+4*1)=5
          d(t2)[1,0]=(2*1+5*1)=7; d(t2)[1,1]=7
          d(t2)[2,0]=(3*1+6*1)=9; d(t2)[2,1]=9
  
    *)
    let expected_t1_grad = N.create [|15.;19.;23.;15.;19.;23.|] [|2;3|] in
    let expected_t2_grad = N.create [|5.;5.;7.;7.;9.;9.|] [|3;2|] in
    assert_bool "t1 grad matmul" (float_ndarray_equal ~epsilon:eps t1.T.grad expected_t1_grad);
    assert_bool "t2 grad matmul" (float_ndarray_equal ~epsilon:eps t2.T.grad expected_t2_grad)
  
  (* Test transpose *)
  (* let test_transpose_backward_multiple _ =
    let t = T.arange 6. in
    let t = T.reshape t ~shape:[|2;3|] in
    let out = T.transpose t in
    out.T.grad <- N.ones [|3;2|];
    (match out.T.backward_fn with Some fn -> fn () | None -> ());
    (* transpose again to verify correct accumulation *)
    out.T.grad <- N.ones [|3;2|];
    (match out.T.backward_fn with Some fn -> fn () | None -> ());
    (* Each backward adds ones transpose again, 
        grad_t after two backward calls should be [|2;3|] of all 2's *)
    let expected = N.full [|2;3|] 2. in
    assert_bool "transpose grad accumulation" (float_ndarray_equal ~epsilon:eps t.T.grad expected)
   *)

  (* Test reshape *)
  let test_reshape_mismatch _ =
    let t = T.arange 8. in
    let out = T.reshape t ~shape:[|2;4|] in
    assert_bool "reshape forward" (shape_equal (N.shape out.T.data) [|2;4|]);
    out.T.grad <- N.ones [|2;4|];
    (match out.T.backward_fn with Some fn -> fn () | None -> ());
    (* Grad should reshape back to original [|8|] *)
    let expected = N.ones [|8|] in
    assert_bool "reshape backward" (float_ndarray_equal ~epsilon:eps t.T.grad expected)
  
  (* Test sum over different dims *)
  let test_sum_no_requires_grad _ =
    let t = T.from_ndarray ~requires_grad:false (N.create [|1.;2.;3.;4.|] [|2;2|]) in
    let out = T.sum ~dim:1 t in
    (* sum along dim 1: [[1+2],[3+4]] = [3,7] *)
    let expected = N.create [|3.;7.|] [|2|] in
    assert_bool "sum forward" (float_ndarray_equal ~epsilon:eps out.T.data expected);
    out.T.grad <- N.ones [|2|];
    (match out.T.backward_fn with Some fn -> fn () | None -> ());
    (* t doesn't require grad, so no grad accumulation *)
    let expected_grad_t = N.zeros [|2;2|] in
    assert_bool "no grad for sum input" (float_ndarray_equal ~epsilon:eps t.T.grad expected_grad_t)
  
  (* Test sum multiple dims *)
  let test_sum_multiple_dims _ =
    let t = T.from_ndarray (N.create [|1.;2.;3.;4.;5.;6.|] [|2;3|]) in
    (* sum along dim=0 *)
    let out = T.sum ~dim:0 t in
    (* sum along dim 0: [[1+4],[2+5],[3+6]] = [5,7,9] -> shape [3] *)
    let expected = N.create [|5.;7.;9.|] [|3|] in
    assert_bool "sum along dim 0" (float_ndarray_equal ~epsilon:eps out.T.data expected);
    out.T.grad <- N.ones [|3|];
    (match out.T.backward_fn with Some fn -> fn () | None -> ());
    (* This should broadcast back along dim 0:
        Original shape: [2;3]
        After sum dim 0, we got shape [3]
        Gradient should expand to two rows:
        [[1,1,1],
        [1,1,1]]
    *)
    let expected_grad = N.ones [|2;3|] in
    assert_bool "sum backward dim 0" (float_ndarray_equal ~epsilon:eps t.T.grad expected_grad)
  
  (* Test mean with different dims *)
  let test_mean_dim _ =
    let t = T.from_ndarray (N.create [|1.;2.;3.;4.;5.;6.|] [|2;3|]) in
    let out = T.mean ~dim:1 t in
    (* mean dim 1: row means -> [ (1+2+3)/3, (4+5+6)/3 ] = [2,5] *)
    let expected = N.create [|2.;5.|] [|2|] in
    assert_bool "mean forward" (float_ndarray_equal ~epsilon:eps out.T.data expected);
    out.T.grad <- N.create [|2.;1.|] [|2|];
    (match out.T.backward_fn with Some fn -> fn () | None -> ());
    (* Backward:
        grad_input = grad_output / 3 broadcasted:
        [[2/3,2/3,2/3],[1/3,1/3,1/3]]
    *)
    let expected_grad = N.create [|(2./.3.);(2./.3.);(2./.3.);(1./.3.);(1./.3.);(1./.3.)|] [|2;3|] in
    assert_bool "mean backward" (float_ndarray_equal ~epsilon:eps t.T.grad expected_grad)
  
  (* Test neg *)
  let test_neg_no_requires_grad _ =
    let t = T.from_ndarray ~requires_grad:false (N.create [|1.;-2.|] [|2|]) in
    let out = T.neg t in
    let expected = N.create [|-1.;2.|] [|2|] in
    assert_bool "neg forward" (float_ndarray_equal ~epsilon:eps out.T.data expected);
    out.T.grad <- N.create [|1.;1.|] [|2|];
    (match out.T.backward_fn with Some fn -> fn () | None -> ());
    (* t does not require grad, so no grad update *)
    let expected_grad = N.zeros [|2|] in
    assert_bool "no grad for neg input" (float_ndarray_equal ~epsilon:eps t.T.grad expected_grad)
  
  (* Test relu with all negative *)
  let test_relu_all_negative _ =
    let t = T.from_ndarray (N.create [|-5.; -4.|] [|2|]) in
    let out = T.relu t in
    let expected = N.zeros [|2|] in
    assert_bool "relu forward negative"
      (float_ndarray_equal ~epsilon:eps out.T.data expected);
    out.T.grad <- N.create [|10.;10.|] [|2|];
    (match out.T.backward_fn with Some fn -> fn () | None -> ());
    (* All negatives result in zero gradient back *)
    let expected_grad = N.zeros [|2|] in
    assert_bool "relu backward all negative"
      (float_ndarray_equal ~epsilon:eps t.T.grad expected_grad)


let float_array_equal ~(epsilon:float) arr1 arr2 =
  let len1 = Array.length arr1 in
  let len2 = Array.length arr2 in
  if len1 <> len2 then false else
    Array.for_alli arr1 ~f:(fun i x ->
      Float.(abs (x -. arr2.(i)) < epsilon))

let float_ndarray_equal ~epsilon nd1 nd2 =
  let arr1 = N.to_array nd1 in
  let arr2 = N.to_array nd2 in
  float_array_equal ~epsilon arr1 arr2


let eps = 1e-7


(* Test softmax *)
let test_softmax _ =
  let t = T.from_ndarray (N.create [|1.; 2.; 3.; 4.|] [|4|]) ~requires_grad:true in
  let out = T.softmax t in
  let max_val = 4. in
  let shifted_logits = [|1. -. max_val; 2. -. max_val; 3. -. max_val; 4. -. max_val|] in
  let exp_logits = Stdlib.Array.map exp shifted_logits in
  let sum_exp = Array.fold ~init:0. ~f:(+.) exp_logits in
  let expected = Stdlib.Array.map (fun x -> x /. sum_exp) exp_logits in
  let expected = N.create expected [|4|] in
  assert_bool "Softmax forward" (float_ndarray_equal ~epsilon:eps out.T.data expected);

  (* Backward *)
  T.zero_grad t; T.zero_grad out;
  out.T.grad <- N.ones [|4|];
  (match out.T.backward_fn with
    | Some fn -> fn ()
    | None -> assert_failure "No backward for softmax");

  (* Check gradient *)
  let softmax_vals = Stdlib.Array.map (fun x -> x /. sum_exp) exp_logits in
  let softmax_nd = N.create softmax_vals [|4|] in
  let sum_grad = N.dsum (N.mul out.T.grad softmax_nd) 0 in
  let expected_grad = N.mul softmax_nd (N.sub out.T.grad sum_grad) in
  assert_bool "Softmax backward"
    (float_ndarray_equal ~epsilon:eps t.T.grad expected_grad)

(* Test log_softmax *)
let test_log_softmax _ =

  let t = T.from_ndarray (N.create [|1.; 2.; 3.; 4.|] [|4|]) ~requires_grad:true in
  let out = T.log_softmax t in
  let max_val = 4. in
  let shifted_logits = [|1. -. max_val; 2. -. max_val; 3. -. max_val; 4. -. max_val|] in
  let sum_exp = Array.fold ~init:0. ~f:(+.) (Stdlib.Array.map Float.exp shifted_logits) in
  let log_probs = Stdlib.Array.map (fun x -> x -. Float.log sum_exp) shifted_logits in 
  let log_probs = N.create log_probs [|4|] in
  assert_bool "Log-Softmax forward" (float_ndarray_equal ~epsilon:eps out.T.data log_probs);

  (* Backward *)
  T.zero_grad t; T.zero_grad out;
  out.T.grad <- N.ones [|4|];
  (match out.T.backward_fn with
    | Some fn -> fn ()
    | None -> assert_failure "No backward for log_softmax");

  (* Check gradient *)
  let softmax_vals = Stdlib.Array.map (fun x -> Float.exp x /. sum_exp) shifted_logits in
  let softmax_nd = N.create softmax_vals [|4|] in
  let sum_grad = N.dsum (N.mul out.T.grad softmax_nd) 0 in
  let expected_grad = N.sub out.T.grad (N.mul softmax_nd sum_grad) in
  assert_bool "Log-Softmax backward"
    (float_ndarray_equal ~epsilon:eps t.T.grad expected_grad)

  let test_slice _ = 
    let t = T.from_ndarray (N.create [|1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.|] [|2; 4|]) ~requires_grad:true in
    
    (* Correct the slice range: The original range [(0, 1); (1, 3)] is valid, 
        but the actual test might not handle the indices correctly. *)
    let out = T.slice t [(0, 1); (1, 3)] in
    (* Correct the expected output: Extract elements [2; 3] within the range, with shape [1; 2]. *)
    let expected_data = N.create [|2.; 3.|] [|1; 2|] in
  
    (* Verify whether the forward pass output is correct *)
    assert_bool "Slice forward"
      (float_ndarray_equal ~epsilon:eps out.T.data expected_data)

let test_dsum _ =
  (* Define a 2x3 tensor for testing sum along different dimensions *)
  let t = T.from_ndarray (N.create [|1.; 2.; 3.; 4.; 5.; 6.|] [|2;3|]) in

  (* Sum along dimension 0 (rows) -> output shape should be [|3|] *)
  let out_dim0 = T.dsum t 0 in
  let expected_dim0 = N.create [|5.;7.;9.|] [|3|] in
  assert_bool "Sum along dim 0" (float_ndarray_equal ~epsilon:eps out_dim0.T.data expected_dim0);

  (* Backward *)
  T.zero_grad t; T.zero_grad out_dim0;
  out_dim0.T.grad <- N.ones [|3|]; (* gradient is all ones for dim 0 sum *)
  
  (match out_dim0.T.backward_fn with
    | Some fn -> fn ()
    | None -> assert_failure "No backward for dim 0 sum");


  (* Gradient checks:
      For t (original shape [2;3]), the gradient for sum along dim 0 should be the broadcasted gradient,
      and the gradient for sum along dim 1 should be summed over the corresponding dimension. *)
  
  (* For dim 0 (sum along rows), grad is [1.; 1.; 1.] repeated across all rows. *)
  let expected_grad_dim0 = N.create [|1.;1.;1.;1.;1.;1.|] [|2;3|] in
  Printf.printf "66666666666666 grad: %s\n" (Ndarray.to_string t.T.grad);
  assert_bool "Grad t (dim 0 sum)" (float_ndarray_equal ~epsilon:eps t.T.grad expected_grad_dim0);

  Printf.printf "test_dsum passed.\n"
      
let test_dmax _ =
  (* Define a 2x3 tensor for testing max along different dimensions *)
  let t = T.from_ndarray (N.create [|1.; 2.; 3.; 4.; 5.; 6.|] [|2;3|]) in

  (* Max along dimension 0 (rows) -> output shape should be [|3|] *)
  let out_dim0 = T.dmax t 0 in
  let expected_dim0 = N.create [|4.;5.;6.|] [|3|] in
  assert_bool "Max along dim 0" (float_ndarray_equal ~epsilon:eps out_dim0.T.data expected_dim0);

  (* Max along dimension 1 (columns) -> output shape should be [|2|] *)
  let out_dim1 = T.dmax t 1 in
  let expected_dim1 = N.create [|3.;6.|] [|2|] in
  assert_bool "Max along dim 1" (float_ndarray_equal ~epsilon:eps out_dim1.T.data expected_dim1);

  (* Backward *)
  T.zero_grad t; T.zero_grad out_dim0; T.zero_grad out_dim1;
  out_dim0.T.grad <- N.ones [|3|]; (* gradient is all ones for dim 0 max *)
  out_dim1.T.grad <- N.ones [|2|]; (* gradient is all ones for dim 1 max *)

  (match out_dim0.T.backward_fn with
    | Some fn -> fn ()
    | None -> assert_failure "No backward for dim 0 max");

  (match out_dim1.T.backward_fn with
    | Some fn -> fn ()
    | None -> assert_failure "No backward for dim 1 max");

  (* Gradient checks:
      For t (original shape [2;3]), the gradient for max along dim 0 should be the broadcasted gradient,
      and the gradient for max along dim 1 should be the maximum value gradient for the corresponding element. *)

  Printf.printf "test_dmax passed.\n"

let test_exp _ =
  (* Define a 2x3 tensor for testing exp *)
  let t = T.from_ndarray (N.create [|1.; 2.; 3.; 4.; 5.; 6.|] [|2;3|]) in

  (* Element-wise exp -> output shape should be [|2;3|] *)
  let out = T.exp t in
  let expected = N.create [|exp 1.; exp 2.; exp 3.; exp 4.; exp 5.; exp 6.|] [|2;3|] in
  assert_bool "Element-wise exp" (float_ndarray_equal ~epsilon:eps out.T.data expected);

  (* Backward *)
  T.zero_grad t; T.zero_grad out;
  out.T.grad <- N.ones [|2;3|]; (* gradient is all ones for exp *)

  (match out.T.backward_fn with
    | Some fn -> fn ()
    | None -> assert_failure "No backward for exp");

  (* Gradient check:
      For exp, the gradient should be the exp of each element: grad = exp(t). *)
  let expected_grad = N.create [|exp 1.; exp 2.; exp 3.; exp 4.; exp 5.; exp 6.|] [|2;3|] in
  assert_bool "Grad t (exp)" (float_ndarray_equal ~epsilon:eps t.T.grad expected_grad);

  Printf.printf "test_exp passed.\n"
  
let test_conv2d _ =
  (* Define a 2x3 input tensor *)
  let input = T.from_ndarray (N.create [|1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.|] [|1; 1; 3; 3|]) in
  (* Define a 1x1 kernel tensor *)
  let kernel = T.from_ndarray (N.create [|1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0|] [|1; 1; 3; 3|]) in

  (* Perform 2D convolution with stride 1 and padding 0 *)
  let output = T.conv2d input kernel ~stride:1 ~padding:0 in
  let expected_output = N.create [| 45.0 |] [|1; 1; 1; 1|] in
  assert_bool "Conv2D forward pass" (float_ndarray_equal ~epsilon:eps output.T.data expected_output);

  (* Backward pass *)
  T.zero_grad input; T.zero_grad kernel; T.zero_grad output;
  output.T.grad <- N.ones [|1; 1; 1; 1|]; (* gradient is all ones for output *)
  Printf.printf "test_conv2d passed.\n"
    

let test_meanpool2d _ =
  (* Define a 1x1x4x4 input tensor *)
  let input = T.from_ndarray (N.create [|1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12.; 13.; 14.; 15.; 16.|] [|1; 1; 4; 4|]) in
  (* Define a 2x2 kernel *)
  let kernel_size = 2 in
  let stride = 2 in

  (* Perform mean pooling with stride 2 and kernel size 2 *)
  let output = T.meanpool2d input ~kernel_size ~stride in
  let expected_output = N.create [|3.5; 5.5; 11.5; 13.5|] [|1; 1; 2; 2|] in
  assert_bool "Meanpool2d forward pass" (float_ndarray_equal ~epsilon:eps output.T.data expected_output);

  (* Backward pass *)
  T.zero_grad input; T.zero_grad output;
  output.T.grad <- N.ones [|1; 1; 2; 2|]; (* gradient is all ones for output *)

  (match output.T.backward_fn with
    | Some fn -> fn ()
    | None -> assert_failure "No backward function for meanpool2d");

  (* Check gradients: 
      - The gradient for each input element should be evenly distributed across the pooled region.
      - Since grad_output is all ones, the input grad should be 0.25 for each element in the pooled region. *)

  let expected_grad_input = N.create [|0.25; 0.25; 0.25; 0.25; 0.25; 0.25; 0.25; 0.25; 0.25; 0.25; 0.25; 0.25; 0.25; 0.25; 0.25; 0.25|] [|1; 1; 4; 4|] in
  assert_bool "Grad input for meanpool2d" (float_ndarray_equal ~epsilon:eps input.T.grad expected_grad_input);

  Printf.printf "test_meanpool2d passed.\n"
let test_tensor_to_string _ =
  let data = N.create [|1.; 2.; 3.; 4.|] [|2;2|] in
  let t = T.{ data=data; requires_grad = true; grad = N.zeros [|2;2|]; backward_fn = None; prev = [] } in
  let result = T.to_string t in
  print_endline result  (* Should print something like: Tensor {data = [[1.0; 2.0]; [3.0; 4.0]], requires_grad = Yes} *)
let suite =
  "Test Tensor Broadcast" >::: [
    "test_broadcast_add" >:: test_broadcast_add;
    "test_broadcast_mul" >:: test_broadcast_mul;
    "test_broadcast_sub" >:: test_broadcast_sub;
    "test_broadcast_div" >:: test_broadcast_div;
    "test_create_basic" >:: test_create_basic;
    "test_from_ndarray_requires_grad_false" >:: test_from_ndarray_requires_grad_false;
    "test_zeros_shape" >:: test_zeros_shape;
    "test_ones_shape" >:: test_ones_shape;
    "test_rand_shape" >:: test_rand_shape;
    "test_xavier_init_shape" >:: test_xavier_init_shape;
    "test_he_init_shape" >:: test_he_init_shape;
    "test_get_set_values" >:: test_get_set_values;
    "test_add_grad_accumulate" >:: test_add_grad_accumulate;
    "test_matmul_non_square" >:: test_matmul_non_square;
    "test_reshape_mismatch" >:: test_reshape_mismatch;
    "test_sum_no_requires_grad" >:: test_sum_no_requires_grad;
    "test_sum_multiple_dims" >:: test_sum_multiple_dims;
    "test_mean_dim" >:: test_mean_dim;
    "test_neg_no_requires_grad" >:: test_neg_no_requires_grad;
    "test_relu_all_negative" >:: test_relu_all_negative;
    "test_softmax" >:: test_softmax;
    "test_log_softmax" >:: test_log_softmax;
    "test_slice" >:: test_slice;
    "test_dsum" >:: test_dsum;
    "test_dmax" >:: test_dmax;
    "test_exp" >:: test_exp;
    "test_conv2d" >:: test_conv2d;
    "test_tensor_to_string" >:: test_tensor_to_string;
    "test_meanpool2d" >:: test_meanpool2d;
  ]

let () =
  run_test_tt_main suite
