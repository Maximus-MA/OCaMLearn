(* test/tests.ml *)

(* Import the module to be tested *)
open Ndarray

(* Helper function to print the shape *)
let print_shape shape =
  Printf.printf "[%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int shape)))
;;

(* Helper function to print the data *)
let print_data data =
  Printf.printf "[%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_float data)))
;;

(* Simple assertion function for floating-point numbers *)
let assert_equal_float expected actual test_name =
  if expected = actual then
    Printf.printf "Passed: %s\n" test_name
  else
    Printf.printf "Failed: %s (expected %f, got %f)\n" test_name expected actual
;;

(* Simple assertion function for floating-point arrays *)
let assert_equal_float_array (expected : float array) (actual : float array) test_name =
  let len_expected = Array.length expected in
  let len_actual = Array.length actual in
  Printf.printf "expeted length %d, actual length %d\n" len_expected len_actual;
  if len_expected <> len_actual then
    Printf.printf "Failed: %s (array lengths differ)\n" test_name
  else
    let passed = ref true in
    let eps = 1e-6 in 
    for i = 0 to len_expected - 1 do
      if abs_float(expected.(i) -. actual.(i)) > eps then
        passed := false
    done;
    if !passed then
      Printf.printf "Passed: %s\n" test_name
    else
      let expected_str = String.concat "; " (Array.to_list (Array.map string_of_float expected)) in
      let actual_str = String.concat "; " (Array.to_list (Array.map string_of_float actual)) in
      Printf.printf "Failed: %s (expected [%s], got [%s])\n" test_name expected_str actual_str
;;

(* Simple assertion function for integer arrays *)
let assert_equal_int_array expected actual test_name =
  let len_expected = Array.length expected in
  let len_actual = Array.length actual in
  if len_expected <> len_actual then
    Printf.printf "Failed: %s (array lengths differ)\n" test_name
  else
    let passed = ref true in
    for i = 0 to len_expected - 1 do
      if expected.(i) <> actual.(i) then
        passed := false
    done;
    if !passed then
      Printf.printf "Passed: %s\n" test_name
    else
      let expected_str = String.concat "; " (Array.to_list (Array.map string_of_int expected)) in
      let actual_str = String.concat "; " (Array.to_list (Array.map string_of_int actual)) in
      Printf.printf "Failed: %s (expected [%s], got [%s])\n" test_name expected_str actual_str
;;

(* Test the set function *)
let test_set () =
  Printf.printf "Testing set function...\n";

  (* Initialize a 2x3 ndarray *)
  let data = [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] in
  let shape = [|2; 3|] in
  let t = create data shape in

  (* Update some elements *)
  let idx1 = [|0; 1|] in
  let idx2 = [|1; 2|] in
  let idx3 = [|0; 0|] in

  set t idx1 10.0;  (* Update value at position [0, 1] to 10.0 *)
  Printf.printf "Finish 1";
  set t idx2 20.0;  (* Update value at position [1, 2] to 20.0 *)
  Printf.printf "Finish 2";
  set t idx3 30.0;  (* Update value at position [0, 0] to 30.0 *)
  Printf.printf "Finish 3";

  (* Check updated data *)
  let expected_data = [|30.0; 10.0; 3.0; 4.0; 5.0; 20.0|] in

  Printf.printf "Updated ndarray data: ";
  print_data t.data;

  (* Verify the result *)
  assert_equal_float_array expected_data t.data "Set Function Data";
  Printf.printf "Passed: Set Function Data\n";

  (* Verify shape remains unchanged *)
  assert_equal_int_array shape t.shape "Set Function Shape";
  Printf.printf "Passed: Set Function Shape\n";
;;

(* Test the at function *)
let test_at () =
  Printf.printf "Testing at function...\n";

  (* Initialize a 3x2x4 ndarray *)
  let data = [|
    1.0; 2.0; 3.0; 4.0;
    5.0; 6.0; 7.0; 8.0;
    9.0; 10.0; 11.0; 12.0;

    13.0; 14.0; 15.0; 16.0;
    17.0; 18.0; 19.0; 20.0;
    21.0; 22.0; 23.0; 24.0
  |] in
  let shape = [|3; 2; 4|] in
  let arr = create data shape in

  (* Test Case 1: Access the first element *)
  let indices1 = [|0; 0; 0|] in
  let value1 = at arr indices1 in
  assert_equal_float 1.0 value1 "At Function - Case 1";

  (* Test Case 2: Access a middle element *)
  let indices2 = [|1; 0; 3|] in
  let value2 = at arr indices2 in
  assert_equal_float 12.0 value2 "At Function - Case 2";

  (* Test Case 3: Access the last element *)
  let indices3 = [|2; 1; 3|] in
  let value3 = at arr indices3 in
  assert_equal_float 24.0 value3 "At Function - Case 3";
;;

(* Test the add function *)
let test_add () =
  Printf.printf "Testing add function...\n";

  (* Test Case 1: Add scalar and vector *)
  let a = create [|2.0|] [|1|] in
  let b = create [|1.0; 2.0; 3.0|] [|3|] in
  let c = add a b in
  Printf.printf "T 1 - R: ";
  print_shape c.shape;
  Printf.printf "Test C 1 - Result data: ";
  print_data c.data;
  let expected_shape = [|3|] in
  let expected_data = [|3.0; 4.0; 5.0|] in
  assert_equal_float_array expected_data c.data "Add Scalar + Vector";
  assert_equal_int_array expected_shape c.shape "Add Scalar + Vector Shape";

  (* Test Case 2: Add matrix and vector *)
  let a = create [|1.0; 2.0; 3.0; 4.0|] [|2; 2|] in
  let b = create [|10.0; 20.0|] [|2|] in
  let c = add a b in
  Printf.printf "Test Case 2 - Result : ";
  print_shape c.shape;
  Printf.printf "Test Case 2 - Result data: ";
  print_data c.data;
  let expected_shape2 = [|2; 2|] in
  let expected_data2 = [|11.0; 22.0; 13.0; 24.0|] in
  assert_equal_float_array expected_data2 c.data "Add Matrix + Vector";
  assert_equal_int_array expected_shape2 c.shape "Add Matrix + Vector Shape";

  (* Test Case 3: Add matrix and scalar *)
  let a = create [|1.0; 2.0; 3.0; 4.0|] [|2; 2|] in
  let b = create [|5.0|] [|1|] in
  let c = add a b in
  Printf.printf "Test Case 3 - Result shape: ";
  print_shape c.shape;
  Printf.printf "Test Case 3 - Result data: ";
  print_data c.data;
  let expected_shape3 = [|2; 2|] in
  let expected_data3 = [|6.0; 7.0; 8.0; 9.0|] in
  assert_equal_float_array expected_data3 c.data "Add Matrix + Scalar";
  assert_equal_int_array expected_shape3 c.shape "Add Matrix + Scalar Shape";

  (* Test Case 4: Add matrix and matrix *)
  let a = create [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] [|2; 1; 3|] in
  let b = create [|10.0; 20.0; 30.0; 40.0|] [|1; 4; 1|] in
  let c = add a b in
  Printf.printf "Test Case 4 - Result shape: ";
  print_shape c.shape;
  Printf.printf "Test Case 4 - Result data: ";
  print_data c.data;
  let expected_shape3 = [|2; 4; 3|] in
  let expected_data3 = [|
    11.0; 12.0; 13.0;
    21.0; 22.0; 23.0;
    31.0; 32.0; 33.0;
    41.0; 42.0; 43.0;
    14.0; 15.0; 16.0;
    24.0; 25.0; 26.0;
    34.0; 35.0; 36.0;
    44.0; 45.0; 46.0;
  |] in
  assert_equal_float_array expected_data3 c.data "Add Matrix + Matrix";
  assert_equal_int_array expected_shape3 c.shape "Add Matrix + Matrix Shape";
;;

(* Test the sub function *)
let test_sub () =
  Printf.printf "Testing sub function...\n";

  (* Test Case 1: Scalar minus vector *)
  let a = { data = [|5.0|]; shape = [|1|] } in
  let b = { data = [|1.0; 2.0; 3.0|]; shape = [|3|] } in
  let c = sub a b in
  Printf.printf "Test Case 1 - Result shape: ";
  print_shape c.shape;
  Printf.printf "Test Case 1 - Result data: ";
  print_data c.data;
  let expected_shape = [|3|] in
  let expected_data = [|4.0; 3.0; 2.0|] in
  assert_equal_float_array expected_data c.data "Sub Scalar - Vector";
  assert_equal_int_array expected_shape c.shape "Sub Scalar - Vector Shape";

  (* Test Case 2: Matrix minus vector *)
  let a = { data = [|5.0; 6.0; 7.0; 8.0|]; shape = [|2; 2|] } in
  let b = { data = [|1.0; 2.0|]; shape = [|2|] } in
  let c = sub a b in
  Printf.printf "Test Case 2 - Result shape: ";
  print_shape c.shape;
  Printf.printf "Test Case 2 - Result data: ";
  print_data c.data;
  let expected_shape2 = [|2; 2|] in
  let expected_data2 = [|4.0; 4.0; 6.0; 6.0|] in
  assert_equal_float_array expected_data2 c.data "Sub Matrix - Vector";
  assert_equal_int_array expected_shape2 c.shape "Sub Matrix - Vector Shape";

  (* Test Case 3: Matrix minus scalar *)
  let a = { data = [|5.0; 6.0; 7.0; 8.0|]; shape = [|2; 2|] } in
  let b = { data = [|3.0|]; shape = [|1|] } in
  let c = sub a b in
  Printf.printf "Test Case 3 - Result shape: ";
  print_shape c.shape;
  Printf.printf "Test Case 3 - Result data: ";
  print_data c.data;
  let expected_shape3 = [|2; 2|] in
  let expected_data3 = [|2.0; 3.0; 4.0; 5.0|] in
  assert_equal_float_array expected_data3 c.data "Sub Matrix - Scalar";
  assert_equal_int_array expected_shape3 c.shape "Sub Matrix - Scalar Shape";

  (* Test Case 4: Matrix minus matrix *)
  let a = { data = [|5.0; 6.0; 7.0; 8.0; 9.0; 10.0|]; shape = [|2; 1; 3|] } in
  let b = { data = [|2.0; 3.0; 4.0; 5.0|]; shape = [|1; 4; 1|] } in
  let c = sub a b in
  Printf.printf "Test Case 4 - Result shape: ";
  print_shape c.shape;
  Printf.printf "Test Case 4 - Result data: ";
  print_data c.data;
  let expected_shape4 = [|2; 4; 3|] in
  let expected_data4 = [|
    3.0; 4.0; 5.0;
    2.0; 3.0; 4.0;
    1.0; 2.0; 3.0;
    0.0; 1.0; 2.0;
    6.0; 7.0; 8.0;
    5.0; 6.0; 7.0;
    4.0; 5.0; 6.0;
    3.0; 4.0; 5.0;
  |] in
  assert_equal_float_array expected_data4 c.data "Sub Matrix - Matrix";
  assert_equal_int_array expected_shape4 c.shape "Sub Matrix - Matrix Shape";

;;

(* Test the mul function *)
let test_mul () =
  Printf.printf "Testing mul function...\n";

  (* Test Case 1: Scalar multiplied by vector *)
  let a = { data = [|2.0|]; shape = [|1|] } in
  let b = { data = [|1.0; 2.0; 3.0|]; shape = [|3|] } in
  let c = mul a b in
  Printf.printf "Test Case 1 - Result shape: ";
  print_shape c.shape;
  Printf.printf "Test Case 1 - Result data: ";
  print_data c.data;
  let expected_shape = [|3|] in
  let expected_data = [|2.0; 4.0; 6.0|] in
  assert_equal_float_array expected_data c.data "Mul Scalar * Vector";
  assert_equal_int_array expected_shape c.shape "Mul Scalar * Vector Shape";

  (* Test Case 2: Matrix multiplied by vector *)
  let a = { data = [|1.0; 2.0; 3.0; 4.0|]; shape = [|2; 2|] } in
  let b = { data = [|10.0; 20.0|]; shape = [|2|] } in
  let c = mul a b in
  Printf.printf "Test Case 2 - Result shape: ";
  print_shape c.shape;
  Printf.printf "Test Case 2 - Result data: ";
  print_data c.data;
  let expected_shape2 = [|2; 2|] in
  let expected_data2 = [|10.0; 40.0; 30.0; 80.0|] in
  assert_equal_float_array expected_data2 c.data "Mul Matrix * Vector";
  assert_equal_int_array expected_shape2 c.shape "Mul Matrix * Vector Shape";

  (* Test Case 3: Matrix multiplied by scalar *)
  let a = { data = [|1.0; 2.0; 3.0; 4.0|]; shape = [|2; 2|] } in
  let b = { data = [|5.0|]; shape = [|1|] } in
  let c = mul a b in
  Printf.printf "Test Case 3 - Result shape: ";
  print_shape c.shape;
  Printf.printf "Test Case 3 - Result data: ";
  print_data c.data;
  let expected_shape3 = [|2; 2|] in
  let expected_data3 = [|5.0; 10.0; 15.0; 20.0|] in
  assert_equal_float_array expected_data3 c.data "Mul Matrix * Scalar";
  assert_equal_int_array expected_shape3 c.shape "Mul Matrix * Scalar Shape";

  (* Test Case 4: Matrix multiplied by matrix *)
  let a = { data = [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|]; shape = [|2; 1; 3|] } in
  let b = { data = [|2.0; 3.0; 4.0; 5.0|]; shape = [|1; 4; 1|] } in
  let c = mul a b in
  Printf.printf "Test Case 4 - Result shape: ";
  print_shape c.shape;
  Printf.printf "Test Case 4 - Result data: ";
  print_data c.data;
  let expected_shape4 = [|2; 4; 3|] in
  let expected_data4 = [|
    2.0; 4.0; 6.0;
    3.0; 6.0; 9.0;
    4.0; 8.0; 12.0;
    5.0; 10.0; 15.0;
    8.0; 10.0; 12.0;
    12.0; 15.0; 18.0;
    16.0; 20.0; 24.0;
    20.0; 25.0; 30.0;
  |] in
  assert_equal_float_array expected_data4 c.data "Mul Matrix * Matrix";
  assert_equal_int_array expected_shape4 c.shape "Mul Matrix * Matrix Shape";

;;

(* Test the div function *)
let test_div () =
  Printf.printf "Testing div function...\n";

  (* Test Case 1: Scalar divided by vector *)
  let a = { data = [|10.0|]; shape = [|1|] } in
  let b = { data = [|2.0; 5.0; 10.0|]; shape = [|3|] } in
  let c = div a b in
  Printf.printf "Test Case 1 - Result shape: ";
  print_shape c.shape;
  Printf.printf "Test Case 1 - Result data: ";
  print_data c.data;
  let expected_shape = [|3|] in
  let expected_data = [|5.0; 2.0; 1.0|] in
  assert_equal_float_array expected_data c.data "Div Scalar / Vector";
  assert_equal_int_array expected_shape c.shape "Div Scalar / Vector Shape";

  (* Test Case 2: Matrix divided by vector *)
  let a = { data = [|10.0; 20.0; 30.0; 40.0|]; shape = [|2; 2|] } in
  let b = { data = [|2.0; 5.0|]; shape = [|2|] } in
  let c = div a b in
  Printf.printf "Test Case 2 - Result shape: ";
  print_shape c.shape;
  Printf.printf "Test Case 2 - Result data: ";
  print_data c.data;
  let expected_shape2 = [|2; 2|] in
  let expected_data2 = [|5.0; 4.0; 15.0; 8.0|] in
  assert_equal_float_array expected_data2 c.data "Div Matrix / Vector";
  assert_equal_int_array expected_shape2 c.shape "Div Matrix / Vector Shape";

  (* Test Case 3: Matrix divided by scalar *)
  let a = { data = [|10.0; 20.0; 30.0; 40.0|]; shape = [|2; 2|] } in
  let b = { data = [|2.0|]; shape = [|1|] } in
  let c = div a b in
  Printf.printf "Test Case 3 - Result shape: ";
  print_shape c.shape;
  Printf.printf "Test Case 3 - Result data: ";
  print_data c.data;
  let expected_shape3 = [|2; 2|] in
  let expected_data3 = [|5.0; 10.0; 15.0; 20.0|] in
  assert_equal_float_array expected_data3 c.data "Div Matrix / Scalar";
  assert_equal_int_array expected_shape3 c.shape "Div Matrix / Scalar Shape";

  (* Test Case 4: Matrix divided by matrix *)
  let a = { data = [|10.0; 20.0; 30.0; 40.0; 50.0; 60.0|]; shape = [|2; 1; 3|] } in
  let b = { data = [|2.0; 4.0; 5.0; 6.0|]; shape = [|1; 4; 1|] } in
  let c = div a b in
  Printf.printf "Test Case 4 - Result shape: ";
  print_shape c.shape;
  Printf.printf "Test Case 4 - Result data: ";
  print_data c.data;
  let expected_shape4 = [|2; 4; 3|] in
  let expected_data4 = [|
    5.0; 10.0; 15.0;
    2.5; 5.0; 7.5;
    2.0; 4.0; 6.0;
    1.6666667; 3.3333333; 5.0;
    20.0; 25.0; 30.0;
    10.0; 12.5; 15.0;
    8.0; 10.0; 12.0;
    6.6666667; 8.3333333; 10.0;
  |] in
  assert_equal_float_array expected_data4 c.data "Div Matrix / Matrix";
  assert_equal_int_array expected_shape4 c.shape "Div Matrix / Matrix Shape";

;;

(* Test the sum_multiplied function *)
let test_sum_multiplied () =
  Printf.printf "Testing sum_multiplied function...\n";

  (* Test Case 1 *)
  let a = [|1.0; 2.0; 3.0|] in
  let b = [|4.0; 5.0; 6.0|] in
  let expected = 32.0 in  (* 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32 *)
  let result = sum_multiplied a b in
  assert_equal_float expected result "sum_multiplied [1.0; 2.0; 3.0] * [4.0; 5.0; 6.0]";

  (* Test Case 2 *)
  let a = [|0.0; 0.0; 0.0|] in
  let b = [|1.0; 2.0; 3.0|] in
  let expected2 = 0.0 in
  let result2 = sum_multiplied a b in
  assert_equal_float expected2 result2 "sum_multiplied [0.0; 0.0; 0.0] * [1.0; 2.0; 3.0]";

  (* Test Case 3 *)
  let a = [|1.5; 2.5; -3.0|] in
  let b = [|4.0; -2.0; 1.0|] in
  let expected3 = 1.5 *. 4.0 +. 2.5 *. (-2.0) +. (-3.0) *. 1.0 in  (* 6.0 -5.0 -3.0 = -2.0 *)
  let result3 = sum_multiplied a b in
  assert_equal_float expected3 result3 "sum_multiplied [1.5; 2.5; -3.0] * [4.0; -2.0; 1.0]";
;;

(* Test the matmul function *)
let test_matmul () =
  Printf.printf "Testing matmul function...\n";

  (* Define matrix A, with shape [batch_size; n] *)
  let batch_size = 2 in  (* b *)
  let n = 3 in
  let cnt = 2 in        (* cnt, used to represent the number of matrices B *)
  let k = 5 in

  let a_data = [|
    1.0; 2.0; 3.0;    (* First sample *)
    4.0; 5.0; 6.0     (* Second sample *)
  |] in
  let a_shape = [|batch_size; n|] in
  let a_ndarray = { data = a_data; shape = a_shape } in

  (* Define matrix B, with shape [cnt; n; k] *)
  let b_data = [|
    (* First matrix, cnt = 0 *)
    1.0; 2.0; 3.0; 4.0; 5.0;    (* n = 0 *)
    6.0; 7.0; 8.0; 9.0; 10.0;   (* n = 1 *)
    11.0;12.0;13.0;14.0;15.0;   (* n = 2 *)

    (* Second matrix, cnt = 1 *)
    2.0; 3.0; 4.0; 5.0; 6.0;    (* n = 0 *)
    7.0; 8.0; 9.0;10.0;11.0;    (* n = 1 *)
    12.0;13.0;14.0;15.0;16.0;   (* n = 2 *)
  |] in
  let b_shape = [|cnt; n; k|] in
  let b_ndarray = { data = b_data; shape = b_shape } in

  (* Compute A * B *)
  let result = matmul a_ndarray b_ndarray in

  (* Print results *)
  Printf.printf "Matrix A shape: ";
  print_shape a_ndarray.shape;
  Printf.printf "Matrix B shape: ";
  print_shape b_ndarray.shape;
  Printf.printf "Result matrix shape: ";
  print_shape result.shape;
  Printf.printf "Result matrix data: ";
  print_data result.data; 

  (* Expected output shape is [batch_size; cnt; k] *)
  let expected_shape = [|batch_size; cnt; k|] in

  (* Manually compute expected results *)
  let expected_data = [|
    46.0; 52.0; 58.0; 64.0; 70.0;
    100.0;115.0;130.0;145.0;160.0;
    52.0; 58.0; 64.0; 70.0; 76.0;
    115.0;130.0;145.0;160.0;175.0;
  |] in

  Printf.printf "Expected matrix data: ";
  print_data expected_data; 
  
  (* Assert Result shape *)
  assert_equal_int_array expected_shape result.shape "Matmul Result Shape";

  (* Assert Result data *)
  let epsilon = 1e-6 in
  let len = Array.length expected_data in
  let passed = ref true in
  for i = 0 to len - 1 do
    if abs_float (expected_data.(i) -. result.data.(i)) > epsilon then
      passed := false
  done;
  if !passed then
    Printf.printf "Passed: Matmul Result Data\n"
  else
    Printf.printf "Failed: Matmul Result Data\n"
;;

(* Test zeros and ones functions *)
let test_zeros_ones () =
  Printf.printf "Testing zeros and ones functions...\n";

  (* Test zeros *)
  let shape = [|2; 3|] in
  let zeros_array = zeros shape in
  Printf.printf "zeros - Result shape: ";
  print_shape zeros_array.shape;
  Printf.printf "zeros - Result data: ";
  print_data zeros_array.data;
  let expected_data = [|0.0; 0.0; 0.0; 0.0; 0.0; 0.0|] in
  assert_equal_float_array expected_data zeros_array.data "Zeros Array Data";
  assert_equal_int_array shape zeros_array.shape "Zeros Array Shape";

  (* Test ones *)
  let ones_array = ones shape in
  Printf.printf "ones - Result shape: ";
  print_shape ones_array.shape;
  Printf.printf "ones - Result data: ";
  print_data ones_array.data;
  let expected_data = [|1.0; 1.0; 1.0; 1.0; 1.0; 1.0|] in
  assert_equal_float_array expected_data ones_array.data "Ones Array Data";
  assert_equal_int_array shape ones_array.shape "Ones Array Shape";
;;

(* Test slice with len = 1 *)
let test_slice_len1 () =
  Printf.printf "Testing slice function for len = 1...\n";

  let data = [|1.0; 2.0; 3.0; 4.0; 5.0|] in
  let shape = [|5|] in
  let nd = create data shape in

  let result = slice nd [(1, 4)] in
  let expected_shape = [|3|] in
  let expected_data = [|2.0; 3.0; 4.0|] in

  Printf.printf "slice len=1 - Result shape: ";
  print_shape result.shape;
  Printf.printf "slice len=1 - Result data: ";
  print_data result.data;

  assert_equal_float_array expected_data result.data "Slice len=1 Array Data";
  assert_equal_int_array expected_shape result.shape "Slice len=1 Array Shape";
;;

(* Test slice with len = 2 *)
let test_slice_len2 () =
  Printf.printf "Testing slice function for len = 2...\n";

  let data = [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0|] in
  let shape = [|2; 4|] in
  let nd = create data shape in

  let result = slice nd [(0, 1); (1, 3)] in
  let expected_shape = [|1; 2|] in
  let expected_data = [|2.0; 3.0|] in

  Printf.printf "slice len=2 - Result shape: ";
  print_shape result.shape;
  Printf.printf "slice len=2 - Result data: ";
  print_data result.data;

  assert_equal_float_array expected_data result.data "Slice len=2 Array Data";
  assert_equal_int_array expected_shape result.shape "Slice len=2 Array Shape";
;;

(* Test slice with len = 3 *)
let test_slice_len3 () =
  Printf.printf "Testing slice function for len = 3...\n";

  let data = [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0; 12.0|] in
  let shape = [|2; 2; 3|] in
  let nd = create data shape in

  let result = slice nd [(0, 2); (0, 1); (1, 3)] in
  let expected_shape = [|2; 1; 2|] in
  let expected_data = [|2.0; 3.0; 8.0; 9.0|] in

  Printf.printf "slice len=3 - Result shape: ";
  print_shape result.shape;
  Printf.printf "slice len=3 - Result data: ";
  print_data result.data;

  assert_equal_float_array expected_data result.data "Slice len=3 Array Data";
  assert_equal_int_array expected_shape result.shape "Slice len=3 Array Shape";
;;

(* Test slice function with len = 4 *)
let test_slice_len4 () =
  Printf.printf "Testing slice function...\n";

  (* Initialize data *)
  let lst = ref [] in 
  for i = 1 to 120 do 
    lst := (float_of_int i) :: !lst 
  done;

  let data = Array.of_list (List.rev !lst) in 
  let shape = [|2; 3; 4; 5|] in 

  (* Create ndarray *)
  let nd = create data shape in 
  
  (* Perform slice operation *)
  let result = slice nd [(0, 1); (1, 3); (1, 4); (3, 5)] in 

  (* Define expected shape and data *)
  let expected_shape = [|1; 2; 3; 2|] in 
  let expected_data = [|29.0; 30.0; 34.0; 35.0; 39.0; 40.0; 49.0; 50.0; 54.0; 55.0; 59.0; 60.0|] in
  
  (* Print the result's shape and data *)
  Printf.printf "slice - Result shape: ";
  print_shape result.shape;
  Printf.printf "slice - Result data: ";
  print_data result.data;

  (* Assert results *)
  assert_equal_float_array expected_data result.data "Slice Array Data";
  assert_equal_int_array expected_shape result.shape "Slice Array Shape";
;;


(* Test sum function *)
let test_sum () =
  Printf.printf "Testing sum function...\n";

  (* Initialize data *)
  let data = [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] in
  let shape = [|2; 3|] in

  (* Create ndarray *)
  let t = create data shape in

  (* Perform sum operation *)
  let result = sum t in
  let expected = 21.0 in

  (* Assert result *)
  assert (result = expected);
  Printf.printf "Passed: Sum Array Data\n";
;;

(* Test mean function *)
let test_mean () =
  Printf.printf "Testing mean function...\n";

  (* Initialize data *)
  let data = [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] in
  let shape = [|2; 3|] in

  (* Create ndarray *)
  let t = create data shape in

  (* Perform mean operation *)
  let result = mean t in
  let expected = 3.5 in

  (* Assert result *)
  assert (result = expected);
  Printf.printf "Passed: Mean Array Data\n";
;;

(* Test variance function *)
let test_var () =
  Printf.printf "Testing variance function...\n";

  (* Initialize data *)
  let data = [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] in
  let shape = [|2; 3|] in

  (* Create ndarray *)
  let t = create data shape in

  (* Perform variance operation *)
  let result = var t in
  let expected = 2.91666667 in

  (* Assert result *)
  assert (abs_float (result -. expected) < 0.00001);
  Printf.printf "Passed: Variance Array Data\n";
;;

(* Test standard deviation function *)
let test_std () =
  Printf.printf "Testing standard deviation function...\n";

  (* Initialize data *)
  let data = [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] in
  let shape = [|2; 3|] in

  (* Create ndarray *)
  let t = create data shape in

  (* Perform standard deviation operation *)
  let result = std t in
  let expected = 1.70782513 in

  (* Assert result *)
  assert (abs_float (result -. expected) < 0.00001);
  Printf.printf "Passed: Standard Deviation Array Data\n";
;;

(* Test max function *)
let test_max () =
  Printf.printf "Testing max function...\n";

  (* Initialize data *)
  let data = [|1.0; 3.0; 5.0; 2.0; 4.0; 6.0|] in
  let shape = [|2; 3|] in

  (* Create ndarray *)
  let t = create data shape in

  (* Perform max operation *)
  let result = max t in
  let expected = 6.0 in

  (* Assert result *)
  assert (result = expected);
  Printf.printf "Passed: Max Array Data\n";
;;

(* Test min function *)
let test_min () =
  Printf.printf "Testing min function...\n";

  (* Initialize data *)
  let data = [|1.0; 3.0; 5.0; 2.0; 4.0; 6.0|] in
  let shape = [|2; 3|] in

  (* Create ndarray *)
  let t = create data shape in

  (* Perform min operation *)
  let result = min t in
  let expected = 1.0 in

  (* Assert result *)
  assert (result = expected);
  Printf.printf "Passed: Min Array Data\n";
;;

(* Test argmax function *)
let test_argmax () =
  Printf.printf "Testing argmax function...\n";

  (* Initialize data *)
  let data = [|1.0; 3.0; 5.0; 2.0; 4.0; 6.0|] in
  let shape = [|2; 3|] in

  (* Create ndarray *)
  let t = create data shape in

  (* Perform argmax operation *)
  let result = argmax t in
  let expected = 5 in

  (* Assert result *)
  assert (result = expected);
  Printf.printf "Passed: Argmax Array Data\n";
;;

(* Test argmin function *)
let test_argmin () =
  Printf.printf "Testing argmin function...\n";

  (* Initialize data *)
  let data = [|1.0; 3.0; 5.0; 2.0; 4.0; 6.0|] in
  let shape = [|2; 3|] in

  (* Create ndarray *)
  let t = create data shape in

  (* Perform argmin operation *)
  let result = argmin t in
  let expected = 0 in

  (* Assert result *)
  assert (result = expected);
  Printf.printf "Passed: Argmin Array Data\n";
;;

(* Initialize 4D test data *)
let initialize_4d_test_data () =
  let lst = ref [] in
  for i = 1 to 120 do
    lst := (float_of_int i) :: !lst
  done;
  let data = Array.of_list (List.rev !lst) in
  let shape = [|2; 3; 4; 5|] in
  create data shape
;;

(* Test dsum *)
let test_dsum () =
  Printf.printf "Testing dsum function...\n";

  let t = initialize_4d_test_data () in

  (* Perform summation along dimension 2 *)
  let result = dsum t 2 in
  let expected_data = [|34.0;  38.0;  42.0;  46.0;  50.0; 114.0; 118.0; 122.0; 126.0; 130.0; 194.0; 198.0; 202.0; 206.0;
  210.0; 274.0; 278.0; 282.0; 286.0; 290.0; 354.0; 358.0; 362.0; 366.0; 370.0; 434.0; 438.0; 442.0;
  446.0; 450.0;|] in
  let expected_shape = [|2; 3; 5|] in 
  print_shape result.shape;
  assert_equal_int_array expected_shape result.shape "Dsum Array Shape";
  assert_equal_float_array expected_data result.data "Dsum Array Data";
  Printf.printf "Passed: Dsum Array Data\n";
;;

(* Test dmean *)
let test_dmean () =
  Printf.printf "Testing dmean function...\n";

  let t = initialize_4d_test_data () in

  (* Perform mean calculation along dimension 3 *)
  let result = dmean t 3 in
  let expected_data = [|3.0;   8.0;  13.0;  18.0;  23.0;  28.0;  33.0;  38.0;  43.0;  48.0;  53.0;  58.0;  63.0;  68.0;
  73.0;  78.0;  83.0;  88.0;  93.0;  98.0; 103.0; 108.0; 113.0; 118.0;|] in
  let expected_shape = [|2; 3; 4|] in 
  print_shape result.shape;
  assert_equal_int_array expected_shape result.shape "Dmean Array Shape";
  assert_equal_float_array expected_data result.data "Dmean Array Data";
  Printf.printf "Passed: Dmean Array Data\n";
;;

(* Test dvariance *)
let test_dvar () =
  Printf.printf "Testing dvariance function...\n";

  let t = initialize_4d_test_data () in

  (* Perform variance calculation along dimension 2 *)
  let result = dvar t 2 in
  let expected_data = [|31.25; 31.25; 31.25; 31.25; 31.25; 31.25; 31.25; 31.25; 31.25; 31.25; 31.25; 31.25;
  31.25; 31.25; 31.25; 31.25; 31.25; 31.25; 31.25; 31.25; 31.25; 31.25; 31.25; 31.25;
  31.25; 31.25; 31.25; 31.25; 31.25; 31.25;
  |] in
  let expected_shape = [|2; 3; 5|] in 
  print_shape result.shape;
  assert_equal_int_array expected_shape result.shape "Dvariance Array Shape";
  assert_equal_float_array expected_data result.data "Dvariance Array Data";
  Printf.printf "Passed: Dvariance Array Data\n";
;;

(* Test dstd *)
let test_dstd () =
  Printf.printf "Testing dstd function...\n";

  let t = initialize_4d_test_data () in

  (* Perform standard deviation calculation along dimension 3 *)
  let result = dstd t 3 in
  let expected_data = [|1.41421356; 1.41421356; 1.41421356; 1.41421356; 1.41421356; 1.41421356;
  1.41421356; 1.41421356; 1.41421356; 1.41421356; 1.41421356; 1.41421356;
  1.41421356; 1.41421356; 1.41421356; 1.41421356; 1.41421356; 1.41421356;
  1.41421356; 1.41421356; 1.41421356; 1.41421356; 1.41421356; 1.41421356;|] in
  let expected_shape = [|2; 3; 4|] in 
  print_shape result.shape;
  assert_equal_int_array expected_shape result.shape "Dstd Array Shape";
  assert_equal_float_array expected_data result.data "Dstd Array Data";
  Printf.printf "Passed: Dstd Array Data\n";
;;

(* Test dmax *)
let test_dmax () =
  Printf.printf "Testing dmax function...\n";

  let t = initialize_4d_test_data () in

  (* Find maximum values along dimension 2 *)
  let result = dmax t 2 in
  let expected_data = [|16.0; 17.0; 18.0; 19.0; 20.0; 36.0; 37.0; 38.0; 39.0; 40.0; 56.0; 57.0; 58.0; 59.0; 
  60.0; 76.0; 77.0; 78.0; 79.0; 80.0; 96.0; 97.0; 98.0; 99.0;100.0;116.0;117.0;118.0; 119.0;120.0;|] in
  let expected_shape = [|2; 3; 5|] in 
  print_shape result.shape;
  assert_equal_int_array expected_shape result.shape "Dmax Array Shape";
  assert_equal_float_array expected_data result.data "Dmax Array Data";
  Printf.printf "Passed: Dmax Array Data\n";
;;

(* Test dmin *)
let test_dmin () =
  Printf.printf "Testing dmin function...\n";

  let t = initialize_4d_test_data () in

  (* Find minimum values along dimension 2 *)
  let result = dmin t 2 in
  let expected_data = [|1.0;   2.0;   3.0;   4.0;   5.0;  21.0;  22.0;  23.0;  24.0;  25.0;  41.0;  42.0;  43.0;  44.0;
  45.0;  61.0;  62.0;  63.0;  64.0;  65.0;  81.0;  82.0;  83.0;  84.0;  85.0; 101.0; 102.0; 103.0;
 104.0; 105.0;|] in
 let expected_shape = [|2; 3; 5|] in 
 print_shape result.shape;
 assert_equal_int_array expected_shape result.shape "Dmin Array Shape";
  assert_equal_float_array expected_data result.data "Dmin Array Data";
  Printf.printf "Passed: Dmin Array Data\n";
;;

(* Test exp function *)
let test_exp () =
  Printf.printf "Testing exp function...\n";
  let data = [|0.0; 1.0; -1.0; 2.0|] in
  let shape = [|4|] in
  let t = create data shape in
  let result = Ndarray.exp t in
  let expected_data = [|Stdlib.exp 0.0; Stdlib.exp 1.0; Stdlib.exp (-1.0); Stdlib.exp 2.0|] in
  assert_equal_float_array expected_data result.data "Exp Function Data";
  assert_equal_int_array shape result.shape "Exp Function Shape";
;;

(* Test log function *)
let test_log () =
  Printf.printf "Testing log function...\n";
  let data = [|1.0; Stdlib.exp 1.0; Stdlib.exp 2.0; Stdlib.exp 3.0|] in
  let shape = [|4|] in
  let t = create data shape in
  let result = log t in
  let expected_data = [|0.0; 1.0; 2.0; 3.0|] in
  assert_equal_float_array expected_data result.data "Log Function Data";
  assert_equal_int_array shape result.shape "Log Function Shape";
;;

(* Test sqrt function *)
let test_sqrt () =
  Printf.printf "Testing sqrt function...\n";
  let data = [|0.0; 1.0; 4.0; 9.0|] in
  let shape = [|4|] in
  let t = create data shape in
  let result = sqrt t in
  let expected_data = [|0.0; 1.0; 2.0; 3.0|] in
  assert_equal_float_array expected_data result.data "Sqrt Function Data";
  assert_equal_int_array shape result.shape "Sqrt Function Shape";
;;

(* Test pow function *)
let test_pow () =
  Printf.printf "Testing pow function...\n";
  let data = [|1.0; 2.0; 3.0; 4.0|] in
  let shape = [|4|] in
  let t = create data shape in
  let exponent = 2.0 in
  let result = pow t exponent in
  let expected_data = [|1.0; 4.0; 9.0; 16.0|] in
  assert_equal_float_array expected_data result.data "Pow Function Data";
  assert_equal_int_array shape result.shape "Pow Function Shape";
;;

(* Test expand_dims function *)
let test_expand_dims () =
  Printf.printf "Testing expand_dims function...\n";
  let data = [|1.0; 2.0; 3.0|] in
  let shape = [|3|] in
  let t = create data shape in
  
  (* Expand at dim = 0 *)
  let result0 = expand_dims t 0 in
  let expected_shape0 = [|1; 3|] in
  assert_equal_int_array expected_shape0 result0.shape "Expand Dims at Dim 0 Shape";
  assert_equal_float_array data result0.data "Expand Dims at Dim 0 Data";

  (* Expand at dim = 1 *)
  let result1 = expand_dims t 1 in
  let expected_shape1 = [|3; 1|] in
  assert_equal_int_array expected_shape1 result1.shape "Expand Dims at Dim 1 Shape";
  assert_equal_float_array data result1.data "Expand Dims at Dim 1 Data";

  (* Attempt to expand at invalid dim *)
  try
    let _ = expand_dims t 3 in
    Printf.printf "Failed: Expand Dims at Dim 3 did not raise exception\n"
  with
  | Failure _ -> Printf.printf "Passed: Expand Dims at Dim 3 raised exception\n"
;;

(* Test squeeze function *)
let test_squeeze () =
  Printf.printf "Testing squeeze function...\n";
  let data = [|1.0; 2.0; 3.0|] in

  (* Squeeze shape [|1; 3; 1|] *)
  let shape = [|1; 3; 1|] in
  let t = create data shape in
  let result = squeeze t in
  let expected_shape = [|3|] in
  assert_equal_int_array expected_shape result.shape "Squeeze Function Shape";
  assert_equal_float_array data result.data "Squeeze Function Data";

  (* Squeeze shape [|1; 1; 1|] *)
  let t2 = create [|5.0|] [|1; 1; 1|] in
  let result2 = squeeze t2 in
  let expected_shape2 = [|1|] in
  let expected_data2 = [|5.0|] in
  assert_equal_int_array expected_shape2 result2.shape "Squeeze Function Shape (All Ones)";
  assert_equal_float_array expected_data2 result2.data "Squeeze Function Data (All Ones)";
;;

(* Test map function *)
let test_map () =
  Printf.printf "Testing map function...\n";
  let data = [|1.0; 2.0; 3.0; 4.0|] in
  let shape = [|2; 2|] in
  let t = create data shape in
  let result = map t ~f:(fun x -> x *. 2.0) in
  let expected_data = [|2.0; 4.0; 6.0; 8.0|] in
  assert_equal_float_array expected_data result.data "Map Function Data";
  assert_equal_int_array shape result.shape "Map Function Shape";
;;

(* Test negate function *)
let test_negate () =
  Printf.printf "Testing negate function...\n";
  let data = [|1.0; -2.0; 3.0; -4.0|] in
  let shape = [|2; 2|] in
  let t = create data shape in
  let result = negate t in
  let expected_data = [|-1.0; 2.0; -3.0; 4.0|] in
  assert_equal_float_array expected_data result.data "Negate Function Data";
  assert_equal_int_array shape result.shape "Negate Function Shape";
;;


(* Test reduce_sum_to_shape function *)
let test_reduce_sum_to_shape () =
  Printf.printf "Testing reduce_sum_to_shape function...\n";
  let data = [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] in
  let shape = [|2; 3|] in
  let t = create data shape in
  let target_shape = [|2; 1|] in
  let result = reduce_sum_to_shape t target_shape in
  let expected_data = [|6.0; 15.0|] in
  let expected_shape = [|2|] in
  assert_equal_float_array expected_data result.data "Reduce Sum to Shape Data";
  print_shape result.shape;
  assert_equal_int_array expected_shape result.shape "Reduce Sum to Shape Shape";
;;

(* Test relu function *)
let test_relu () =
  Printf.printf "Testing relu function...\n";
  let data = [|1.0; -2.0; 3.0; -4.0|] in
  let shape = [|2; 2|] in
  let t = create data shape in
  let result = relu t in
  let expected_data = [|1.0; 0.0; 3.0; 0.0|] in
  assert_equal_float_array expected_data result.data "ReLU Function Data";
  assert_equal_int_array shape result.shape "ReLU Function Shape";
;;

(* Test to_string function *)
let test_to_string () =
  Printf.printf "Testing to_string function...\n";
  let data = [|1.0; 2.0; 3.0; 4.0|] in
  let shape = [|2; 2|] in
  let t = create data shape in
  let a = rand shape in 
  let b = xavier_init shape in
  let c = kaiming_init shape in 
  let d = add t a in 
  let e = add b c in 
  let f = add d e in
  let f = transpose f in
  let g = arange 2.0 in 
  let result = add g f in 
  let result = reshape result shape in 
  let out = to_array result in 
  assert (out.(0) <> 1.23);
  let result = to_string result in
  let expected = to_string f in
  (* Printf.printf "%s\n%s" result expected; *)
  assert (result <> expected);
  Printf.printf "Passed: To String Function\n";
;;

(* Test pad_shape_to function *)
let test_pad_shape_to () =
  Printf.printf "Testing pad_shape_to function...\n";

  (* Case 1: arr_shape and target_shape have the same length *)
  let arr_shape1 = [|3; 4; 5|] in
  let target_shape1 = [|3; 4; 5|] in
  let result_shape1, result_padded1 = pad_shape_to arr_shape1 target_shape1 in
  assert_equal_int_array arr_shape1 result_shape1 "Pad Shape To - Same Length (arr_shape)";
  assert_equal_int_array target_shape1 result_padded1 "Pad Shape To - Same Length (target_shape)";
  Printf.printf "Passed: Pad Shape To - Same Length\n";

  (* Case 2: arr_shape is longer than target_shape *)
  let arr_shape2 = [|3; 4; 5|] in
  let target_shape2 = [|5|] in
  let expected_padded2 = [|1; 1; 5|] in
  let result_shape2, result_padded2 = pad_shape_to arr_shape2 target_shape2 in
  assert_equal_int_array arr_shape2 result_shape2 "Pad Shape To - arr_shape Longer (arr_shape)";
  assert_equal_int_array expected_padded2 result_padded2 "Pad Shape To - arr_shape Longer (padded target_shape)";
  Printf.printf "Passed: Pad Shape To - arr_shape Longer\n";

;;
let test_conv2d () =
  (* Define a simple input tensor (batch_size=1, in_channels=2, height=5, width=5) *)
  let input_data = [|
    1.0; 2.0; 3.0; 4.0; 5.0;
    6.0; 7.0; 8.0; 9.0; 10.0;
    11.0; 12.0; 13.0; 14.0; 15.0;
    16.0; 17.0; 18.0; 19.0; 20.0;
    21.0; 22.0; 23.0; 24.0; 25.0;
    1.0; 1.0; 1.0; 1.0; 1.0;
    1.0; 1.0; 1.0; 1.0; 1.0;
    1.0; 1.0; 1.0; 1.0; 1.0;
    1.0; 1.0; 1.0; 1.0; 1.0;
    1.0; 1.0; 1.0; 1.0; 1.0
  |] in
  let input_shape = [| 1; 2; 5; 5 |] in
  let input = Ndarray.create input_data input_shape in

  (* Define a simple kernel (out_channels=2, in_channels=2, height=3, width=3) *)
  let kernel_data = [|
    1.0; 0.0; -1.0;
    1.0; 0.0; -1.0;
    1.0; 0.0; -1.0;
    1.0; 0.0; -1.0;
    1.0; 0.0; -1.0;
    1.0; 0.0; -1.0;
    0.5; 0.5; 0.5;
    0.5; 0.5; 0.5;
    0.5; 0.5; 0.5;
    0.5; 0.5; 0.5;
    0.5; 0.5; 0.5;
    0.5; 0.5; 0.5
  |] in
  let kernel_shape = [| 2; 2; 3; 3 |] in
  let kernel = Ndarray.create kernel_data kernel_shape in

  (* Perform 2D convolution with stride=1 and padding=1 *)
  let stride = 1 in
  let padding = 1 in
  let output = Ndarray.conv2d input kernel stride padding in

  (* Print the output tensor *)
  Printf.printf "Output shape: [%s]\n" (String.concat "; " (Array.to_list (Array.map string_of_int output.shape)));
  Printf.printf "Output data:\n";
  Array.iteri (fun i x ->
    if i > 0 && i mod output.shape.(3) = 0 then Printf.printf "\n";
    Printf.printf "%.1f " x
  ) output.data;
  Printf.printf "\n"

let test_transpose_last_two_dims () =
  (* Define a simple input tensor (batch_size=1, channels=1, height=3, width=4) *)
  let input_data = [|
    1.0; 2.0; 3.0; 4.0;
    5.0; 6.0; 7.0; 8.0;
    9.0; 10.0; 11.0; 12.0;
    1.0; 2.0; 3.0; 4.0;
    5.0; 6.0; 7.0; 8.0;
    9.0; 10.0; 11.0; 12.0
  |] in
  let input_shape = [| 1; 2; 3; 4 |] in
  let input = Ndarray.create input_data input_shape in

  (* Perform transpose of the last two dimensions *)
  let output = Ndarray.transpose_last_two_dims input in

  (* Print the output tensor *)
  Printf.printf "Input shape: [%s]\n" (String.concat "; " (Array.to_list (Array.map string_of_int input.shape)));
  Printf.printf "Input data:\n";
  Array.iteri (fun i x ->
    if i > 0 && i mod input.shape.(3) = 0 then Printf.printf "\n";
    Printf.printf "%.1f " x
  ) input.data;
  Printf.printf "\n\n";

  Printf.printf "Output shape: [%s]\n" (String.concat "; " (Array.to_list (Array.map string_of_int output.shape)));
  Printf.printf "Output data:\n";
  Array.iteri (fun i x ->
    if i > 0 && i mod output.shape.(3) = 0 then Printf.printf "\n";
    Printf.printf "%.1f " x
  ) output.data;
  Printf.printf "\n"
(* Main test function *)
let () =
  Printf.printf "Start Test!";
  test_set();
  test_at ();
  test_add ();
  test_sum_multiplied ();
  test_matmul();
  test_sub ();
  test_mul ();
  test_div ();
  test_add ();
  test_zeros_ones ();
  test_slice_len1 ();
  test_slice_len2 ();
  test_slice_len3 ();
  test_slice_len4 ();
  test_sum ();
  test_mean ();
  test_var ();
  test_std ();
  test_max ();
  test_min ();
  test_argmax ();
  test_argmin ();
  test_dsum ();
  test_dmean ();
  test_dvar ();
  test_dstd ();
  test_dmax ();
  test_dmin ();
  test_exp ();
  test_log ();
  test_sqrt ();
  test_pow ();
  test_expand_dims ();
  test_squeeze ();
  test_map ();
  test_negate ();
  test_reduce_sum_to_shape ();
  test_relu ();
  test_to_string ();
  test_pad_shape_to ();
  test_conv2d ();
  test_transpose_last_two_dims ();
;;
