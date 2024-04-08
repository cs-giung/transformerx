"""This module provides parser combinators for einshard."""
from functools import partial
from operator import itemgetter
from typing import Callable, Generic, TypeVar
import unicodedata


class VoidType: # pylint: disable=too-few-public-methods
    """A class to represent a void type."""
    def __repr__(self):
        return 'Void'

Void = VoidType()

def _void_type_new(cls):
    raise RuntimeError('VoidType cannot be instantiated directly.')

def _void_type_init_subclass(cls, **kwargs):
    raise RuntimeError('Subclassing VoidType is not allowed.')

VoidType.__new__ = _void_type_new
VoidType.__init_subclass__ = _void_type_init_subclass

T = TypeVar('T')
E = TypeVar('E')

class Either(Generic[T, E]): # pylint: disable=too-few-public-methods
    """A generic class representing an either type."""
    def is_success(self) -> bool:
        """it should indicate if the instance represents a success."""
        raise NotImplementedError('Subclasses should implement this method.')

class Success(Either[T, E]):
    """A subclass of Either representing the success case."""
    def __init__(self, value: T):
        self.value = value

    def __repr__(self):
        return f'Success({self.value!r})'

    def is_success(self) -> bool:
        return True

    def get(self) -> T: # pylint: disable=missing-function-docstring
        return self.value

    def __eq__(self, other):
        if not isinstance(other, Success):
            return False
        return self.value == other.value

class Failure(Either[T, E]):
    """A subclass of Either representing the failure case."""
    def __init__(self, error: E):
        self.error = error

    def __repr__(self):
        return f'Failure({self.error!r})'

    def is_success(self) -> bool:
        return False

    def get(self) -> T: # pylint: disable=missing-function-docstring
        return self.error

    def __eq__(self, other):
        if not isinstance(other, Failure):
            return False
        return self.error == other.error

Position = int
ErrorType = str
R = TypeVar('R')
U = TypeVar('U')
V = TypeVar('V')

ParseResultSuccess = tuple[Position, R]
ParseResultFailure = tuple[Position, ErrorType]
ParseResult = Either[ParseResultSuccess, ParseResultFailure]

Parser = Callable[[str, Position], ParseResult[R]]
ParserSuccess = Callable[[str, Position], ParseResultSuccess[R]]
ParserFailure = Callable[[str, Position], ParseResultFailure]

def satisfy(predicate: Callable[[str], bool], desc: str) -> Parser[str]:
    # pylint: disable=missing-function-docstring
    def f(s: str, idx: Position) -> ParseResult[str]:
        if idx == len(s):
            return Failure((idx, desc))
        c = s[idx]
        if not predicate(c):
            return Failure((idx, desc))
        idx += 1
        return Success((idx, c))
    return f

def many(parser: Parser[str]) -> Parser[list[str]]:
    # pylint: disable=missing-function-docstring
    def f(s: str, idx: Position) -> ParseResult[list[str]]:
        out = []
        while True:
            res = parser(s, idx)
            if not res.is_success():
                return Success((idx, out))
            idx, token = res.get()
            out.append(token)
    return f

def many1(parser: Parser[str]) -> Parser[list[str]]:
    # pylint: disable=missing-function-docstring
    def f(s: str, idx: Position) -> ParseResult[list[str]]:
        out = []
        res = parser(s, idx)
        if not res.is_success():
            return res
        idx, token = res.get()
        out.append(token)
        res = many(parser)(s, idx)
        if res.is_success():
            idx, token = res.get()
            out.extend(token)
        return Success((idx, out))
    return f

def literal(s: str, *, desc: str | None = None) -> Parser[str]:
    # pylint: disable=missing-function-docstring
    if desc is None:
        desc = f'string "{s}"'
    len_literal = len(s)
    def f(s_: str, idx: Position) -> ParseResult[str]:
        if idx > len(s_) - len_literal or s_[idx:idx+len_literal] != s:
            return Failure((idx, desc))
        idx += len_literal
        return Success((idx, s))
    return f

def parse_eof(s: str, idx: Position) -> ParseResult[VoidType]:
    # pylint: disable=missing-function-docstring
    if idx != len(s):
        return Failure((idx, 'eof'))
    return Success((idx, Void))

def with_default(parser: Parser[U], *, default: U) -> ParserSuccess[U]:
    # pylint: disable=missing-function-docstring
    def f(s: str, idx: Position) -> ParseResult[U]:
        res = parser(s, idx)
        if res.is_success():
            return res
        return Success((idx, default))
    return f

def with_description(parser: Parser[U], desc: str) -> Parser[U]:
    # pylint: disable=missing-function-docstring
    def f(s: str, idx: Position) -> ParseResult[U]:
        res = parser(s, idx)
        if res.is_success():
            return res
        idx, _ = res.get()
        return Failure((idx, desc))
    return f

def pmap(func: Callable[[U], V], parser: Parser[U]) -> Parser[V]:
    # pylint: disable=missing-function-docstring
    def f(s: str, idx: Position) -> ParseResult[V]:
        res = parser(s, idx)
        if not res.is_success():
            return res
        idx, token = res.get()
        return Success((idx, func(token)))
    return f

def pchain(*parsers: Parser[U]) -> Parser[list[U]]:
    # pylint: disable=missing-function-docstring
    def f(s: str, idx: Position) -> ParseResult[list[U]]:
        out = []
        for parser in parsers:
            res = parser(s, idx)
            if not res.is_success():
                return res
            idx, val = res.get()
            if val is not Void:
                out.append(val)
        return Success((idx, out))
    return f

def pjoin(parser: Parser[list[str]]) -> Parser[str]:
    # pylint: disable=missing-function-docstring
    def f(s: str, idx: Position) -> ParseResult[str]:
        res = parser(s, idx)
        if not res.is_success():
            return res
        idx, token = res.get()
        return Success((idx, ''.join(token)))
    return f

def anyof(*parsers: Parser[U]) -> Parser[U]:
    # pylint: disable=missing-function-docstring
    def _join_descriptions(xs: list[str]) -> str:
        if len(xs) == 0:
            return ''
        if len(xs) == 1:
            return xs
        *init, tail = xs
        return ', '.join(init) + ' or ' + tail
    def f(s: str, idx: Position) -> ParseResult[U]:
        descriptions = []
        for parser in parsers:
            res = parser(s, idx)
            if res.is_success():
                return res
            _, desc = res.get()
            descriptions.append(desc)
        return Failure((idx, _join_descriptions(descriptions)))
    return f

def pselect(idx: Position, parser: Parser[U]) -> Parser[V]:
    # pylint: disable=missing-function-docstring
    return pmap(itemgetter(idx), parser)

sepby1: Callable[[Parser[U], Parser[V]], Parser[U]] = \
    lambda parser, separator: pmap(
        lambda xs: [xs[0], *xs[1]],
        pchain(parser, many(pselect(1, pchain(separator, parser)))))

const: Callable[[U], Callable[[V], U]] = lambda x: lambda _: x
pvoid: Callable[[Parser[U]], Parser[VoidType]] = partial(pmap, const(Void))

is_0_to_9: Callable[[str], bool] = lambda c: c.isdigit()
is_1_to_9: Callable[[str], bool] = lambda c: c.isdigit() and c != '0'
parse_0_to_9 = satisfy(is_0_to_9, 'digit 0-9')
parse_1_to_9 = satisfy(is_1_to_9, 'digit 1-9')

is_identifier_char: Callable[[str], bool] = \
    lambda c: unicodedata.category(c)[0] == 'L' or c == '_'
parse_identifier_char = satisfy(is_identifier_char, 'identifier')
parse_identifier = pjoin(many1(parse_identifier_char))

is_space: Callable[[str], bool] = lambda c: c.isspace()
parse_space = satisfy(is_space, 'space')
parse_spaces = many1(parse_space)
parse_spaces_optional = pvoid(many(parse_space))

parse_right_arrow = pvoid(literal('->'))
parse_ellipsis = pmap(const(...), literal('...', desc='ellipsis'))
parse_integer = pmap(
    int, pjoin(pchain(parse_1_to_9, pjoin(many(parse_0_to_9)))))

parse_element_left = anyof(parse_identifier, parse_ellipsis)
parse_element_right = anyof(
    pchain(with_default(parse_identifier, default=None), parse_integer),
    pchain(parse_identifier, with_default(parse_integer, default=0)),
    parse_ellipsis,
)
parse_elements_left = sepby1(parse_element_left, parse_spaces)
parse_elements_right = sepby1(parse_element_right, parse_spaces)

parse_expression = pchain(
    parse_spaces_optional,
    parse_elements_left,
    parse_spaces_optional,
    parse_right_arrow,
    parse_spaces_optional,
    parse_elements_right,
    parse_spaces_optional,
    parse_eof,
)
