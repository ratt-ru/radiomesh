import argparse


def main():
  parser = argparse.ArgumentParser(
    prog="radiomesh", description="Command-line tool for radiomesh package tasks"
  )

  subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")
  _ = subparsers.add_parser("gen-expr", help="Generate sympy expressions")
  args = parser.parse_args()

  if args.command == "gen-expr":
    from radiomesh.scripts.gen_expr import generate_expression

    generate_expression(args)
  else:
    parser.print_help()


if __name__ == "__main__":
  main()
