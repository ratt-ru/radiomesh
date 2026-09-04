import ast
import re
from typing import List, Literal, Tuple, TypeGuard, get_args

type BaseParseTerm = str | List[str] | Tuple[str, ...]
type ParseTerm = BaseParseTerm | List["BaseParseTerm"] | Tuple["BaseParseTerm", ...]
type EquationTerms = List[str | Tuple[str, ...]]


# Left Antenna, Baseline, Right Antenna
TermLocality = Literal["p", "pq", "q"]


class RimeParseError(ValueError):
  pass


class RimeSpecificationError(ValueError):
  pass


class IterableParser(ast.NodeTransformer):
  """Recursively parses lists or tuples of strings
  or numbers into their python equivalents"""

  def visit_Module(self, node):
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
      raise RimeParseError("Module must contain a single expression")

    if not isinstance((expr := node.body[0]).value, (ast.Tuple, ast.List)):
      raise RimeParseError(f"Expression {expr.value} must be a tuple or list")

    return self.visit(expr).value

  def visit_List(self, node):
    return list(self.visit(v) for v in node.elts)

  def visit_Tuple(self, node):
    return tuple(self.visit(v) for v in node.elts)

  def visit_Name(self, node):
    return node.id

  def visit_Num(self, node):
    return node.value


ITERABLE_PARSER = IterableParser()


def parse_iterable(values: str) -> ParseTerm:
  """Recursively parse lists/tuples of strings/ints"""
  return ITERABLE_PARSER.visit(ast.parse(values))


RIME_EXEMPLAR_STR = "[Gp, (Kpq, Bpq, Gq)]: [I,Q,U,V] -> [XX,XY,YX,YY]"


def _is_str_list(value: object) -> TypeGuard[List[str]]:
  return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _is_valid_rime_term_spec(value: object) -> TypeGuard[List[str | Tuple[str, ...]]]:
  if not isinstance(value, list):
    return False

  for term in value:
    if isinstance(term, tuple):
      if not all(isinstance(v, str) for v in term):
        return False
    elif not isinstance(term, str):
      return False

  return True


def parse_stokes(stokes_str: str) -> List[str]:
  """Parse a list of stokes parameters encapsulated in brackets or parentheses"""
  if not _is_str_list(stokes := parse_iterable(stokes_str)):
    raise RimeParseError(
      f"Stokes specification '{stokes_str}' must be of the form [I,Q,U,V]"
    )

  return [s.upper() for s in stokes]


def parse_pols(pols_str: str) -> List[str]:
  """Parse a list of polarisations parameter encapsulated in brackets or parentheses"""
  if not _is_str_list(pols := parse_iterable(pols_str)):
    raise RimeParseError(
      f"Polarisation specification '{pols_str}' must be of the form [XX,XY,YX,YY]"
    )
  return [p.upper() for p in pols]


def parse_rime(
  rime: str,
) -> Tuple[List[str | Tuple[str, ...]], Tuple[str, ...], Tuple[str, ...]]:
  f"""Parse a rime specification string into terms, stokes parameters and polarisations

  The specification should be of the form {RIME_EXEMPLAR_STR}"""
  if len(bits := [s.strip() for s in rime.split(":")]) != 2:
    raise RimeParseError(f"RIME {rime} is not of the form {RIME_EXEMPLAR_STR}.")

  rime_bits, covert_bits = bits

  if len(bits := [s.strip() for s in covert_bits.split("->")]) != 2:
    raise RimeParseError(
      f"RIME Polarisation conversion specification '{covert_bits}' "
      f"must be of the form [I,Q,U,V] -> [XX,XY,YX,YY]."
    )

  stokes_bits, pol_bits = bits
  stokes = tuple(parse_stokes(stokes_bits))
  pols = tuple(parse_pols(pol_bits))

  if not _is_valid_rime_term_spec(terms := parse_iterable(rime_bits)):
    raise RimeParseError(
      f"RIME equation '{terms}' must be a tuple/list of terms "
      f"of the form [Gp, (Lp, Kpq, Bpq, Lq), Gq]"
    )

  return terms, stokes, pols


# Rime terms of the form Gp, Gq, Kpq where
# the capital letter uniquely identifies the term, while the
# locality (p, pq or q) indicates whether the term is local to
# the left antenna, the baseline or the right antenna, respectively
TERM_STRING_REGEX = re.compile(r"(?P<symbol>[A-Z])(?P<locality>pq|p|q)")


def parse_terms(
  terms: EquationTerms,
) -> Tuple[List[Tuple[str, TermLocality]], List[Tuple[str, TermLocality]], int]:
  """Parse RIME equation terms into
  direction independent and direction dependent terms"""

  def _check_term_ordering(name, terms):
    order = dict((i, lo) for (lo, i) in enumerate(get_args(TermLocality)))

    def cmp(x, y, o):
      return (o[x] > o[y]) - (o[y] - o[x])

    if not all(cmp(x, y, order) <= 0 for (_, x), (_, y) in zip(terms[:-1], terms[1:])):
      raise RimeSpecificationError(
        f"{name} terms {terms} don't satisfy p < pq < q ordering constraint."
      )

  def _handle_term(term, container):
    if not isinstance(term, str):
      raise TypeError(f"{term} is not a string")

    if (m := TERM_STRING_REGEX.match(term)) is None:
      raise RimeSpecificationError(
        f"Term '{term}' is not of the form Gp, Gq or Gpq "
        f"where G is the term symbol and the locality "
        f"(p, q or pq) denotes the left antenna, right antenna "
        f"and baseline pair, respectively."
      )

    symbol = m.group("symbol")
    locality = m.group("locality")
    container.append((symbol, locality))

  die_terms: List[Tuple[str, TermLocality]] = []
  dde_terms: List[Tuple[str, TermLocality]] = []
  dde_index = -1

  for t, term in enumerate(terms):
    if isinstance(term, str):
      _handle_term(term, die_terms)

    elif isinstance(inner_terms := term, (tuple, list)):
      if dde_index != -1:
        raise RimeSpecificationError(f"Multiple DDE term groups were present {terms}")

      for inner_term in inner_terms:
        _handle_term(inner_term, dde_terms)

      dde_index = t
    else:
      raise TypeError(f"Invalid RIME Term {term}")

  _check_term_ordering("DIE", die_terms)
  _check_term_ordering("DDE", dde_terms)

  return die_terms, dde_terms, dde_index


class RimeSpecification:
  VALID_STOKES = {"I", "Q", "U", "V"}
  LINEAR_POLS = ["XX", "XY", "YX", "YY"]
  CIRCULAR_POLS = ["RR", "RL", "LR", "LL"]

  def __init__(self, specification: str):
    self._specification = specification
    terms, self._stokes, self._pols = parse_rime(specification)

    if not set(self._stokes).issubset(self.VALID_STOKES):
      raise RimeSpecificationError(
        f"Stokes {self._stokes} parameters not in {self.VALID_STOKES}"
      )

    self._feed_type = self._infer_feed_type(self._pols)
    self._die_terms, self._dde_terms, self._dde_index = parse_terms(terms)

  def __reduce__(self):
    return (RimeSpecification, (self._specification,))

  def __hash__(self) -> int:
    return hash(
      (
        tuple(self._pols),
        tuple(self._stokes),
        self._feed_type,
        tuple(self._die_terms),
        tuple(self._dde_terms),
        self._dde_index,
      )
    )

  @classmethod
  def _infer_feed_type(cls, pols: Tuple[str, ...]) -> Literal["linear", "circular"]:
    if (pol_set := set(pols)).issubset(cls.LINEAR_POLS):
      return "linear"
    elif pol_set.issubset(cls.CIRCULAR_POLS):
      return "circular"
    else:
      raise RimeSpecificationError(
        f"Polarisations {pols} are not purely linear or circular"
      )
