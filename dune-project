(lang dune 3.16)

(name OCaML)

(generate_opam_files true)



(source
 (github username/reponame))

(authors "Author Name")

(maintainers "Maintainer Name")

(license LICENSE)

(documentation https://url/to/documentation)

(package
 (name OCaML)
 (synopsis "A short synopsis")
 (description "A longer description")
 (depends ocaml dune)
 (tags
  (topics "to describe" your project))
 (allow_empty))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/reference/dune-project/index.html
