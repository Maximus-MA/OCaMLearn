(lang dune 3.16)

(generate_opam_files true)

(authors "Suizhi Ma" "Chuan Chen" "Jixiao Zhang" "Rui Wang")

(maintainers "ulmd06@gmail.com")

(package
  (name OCaMLearn)

  (allow_empty)

  (synopsis "An OCaml library for Machine Learning Framework")

  (description "A machine learning library in OCaml, inspired by PyTorch, to provide the essential components for building and training machine learning models.")

  (depends
    (ounit2 (>= 2.2.7)) ;
    (core (>= 0.17.1)) ; 
    (csv (>= 2.4))
))