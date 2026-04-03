import argparse


def main():
  parser = argparse.ArgumentParser(
    prog="radiomesh", description="Command-line tool for radiomesh package tasks"
  )

  subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")
  _ = subparsers.add_parser("gen-expr", help="Generate sympy expressions")
  _ = subparsers.add_parser(
    "gen-kernel-params",
    help="Download ducc0 KernelDB and "
    "regenerate radiomesh/generated/_es_kernel_params.py",
  )
  args = parser.parse_args()

  if args.command == "gen-expr":
    from radiomesh.scripts.gen_expr import generate_expression

    generate_expression(args)
  elif args.command == "gen-kernel-params":
    from radiomesh.scripts.gen_kernel_params import generate_kernel_params

    generate_kernel_params()
  else:
    parser.print_help()


if __name__ == "__main__":
  main()
