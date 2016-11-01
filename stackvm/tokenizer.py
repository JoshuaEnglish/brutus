"""lexer and tokenizer for StackVM
"""
from __future__ import (absolute_import, print_function)

from string import (whitespace as WHITESPACE,
    digits as DIGITS,
    ascii_letters as LETTERS)

IDENTIFIER_START = LETTERS + '_'
IDENTIFIER = IDENTIFIER_START + DIGITS + "':"

COMMENT_START = '#'

DIGITS += '-+'

EOL = '\n\r'

class Symbol(int):
    """
    Integer value with name for ease of identification
    """

    _next_id = 1

    def __new__(cls, name, id_=None):
        symbol_id = id_ if id_ is not None else cls.next_id()
        self = super(Symbol, cls).__new__(cls, symbol_id)
        self._name = name
        return self

    def __str__(self):
        return '%s (%d)' % (self._name, int(self))

    @property
    def name(self):
        """
        The string name of this symbol, as spelled in the grammar, suitable
        for use in the user interface.
        """
        return self._name

    @classmethod
    def next_id(cls):
        """
        Return the next available symbol identifier for this symbol type.
        """
        id_ = cls._next_id
        cls._next_id += 1
        return id_

    def __repr__(self):
        return "%s(%d, '%s')" % (self.__class__.__name__, int(self), self._name)

class TerminalSymbol(Symbol):
    _next_id = 1
    is_terminal = True

class NonterminalSymbol(Symbol):
    _next_id = 1000
    is_terminal = False


SNTL = Symbol('SNTL', 0)

NAME = Symbol('NAME')

TEXT = Symbol('TEXT')

OPERATOR = Symbol('OPERATOR')
STORAGE = Symbol('STORAGE')
REGISTER = Symbol('REGISTER')

NUMBER = Symbol('NUMBER')

LABEL = Symbol('LABEL')
KEYWORD = Symbol('KEYWORD')

class Token(object):
    """
    A token, classified by a terminal symbol (token class) and containing
    a lexeme of that class.
    """

    __slots__ = ('_symbol', '_lexeme')

    def __init__(self, symbol, lexeme):
        self._symbol = symbol  # always a terminal symbol
        self._lexeme = lexeme

    def __repr__(self):
        """
        Like 'Token(COMMA, ",")'.
        """
        return 'Token(%s, "%s")' % (self._symbol.name, self._lexeme)

    @property
    def symbol(self):
        """
        The terminal symbol this token is an instance of.
        """
        return self._symbol

    @property
    def lexeme(self):
        """
        String value (lexeme) of this token.
        """
        return self._lexeme

    @property
    def value(self):
        """
        The string value (lexeme) of this token. Useful generic alternative
        when walking an AST in which this token is a node. Both ASTNode and
        Token objects have a value property.
        """
        return self._lexeme

class BaseLexer(object):
    def __init__(self, input, start_state='_lex_start', emit_sntl=True):
        self._input = input
        self._start = 0
        self._pos = 0
        self._start_state = getattr(self, start_state)
        self._emit_sntl = emit_sntl
        self._tokens = []
        self.state = None
        self.tokenclass = Token

    def __iter__(self):
        self._start = self._pos = 0
        self.state = self._start_state
        return iter(self._next_token, None)

    def _next_token(self):
        if self._token_in_queue:
            return self._pop_token()

        while self.state is not None:
            self.state = self.state()
            if self._token_in_queue:
                return self._pop_token()

        if self._start != self._pos or self._pos < self._len:
            raise ValueError('not all input consumed')

    def _accept_run(self, charset):
        while True:
            c = self._next()
            if c is None or c not in charset:
                self._backup()
                break
            
    def _accept_until(self, charset):
        while True:
            c = self._next()
            if c is None or c in charset:
                self._backup()
                break

    def _backup(self):
        self._pos -= 1

    def _emit(self, token_type):
        lexeme = self._input[self._start:self._pos]
        self._start = self._pos
        self._tokens.append(self.tokenclass(token_type, lexeme))

    def _ignore(self):
        self._start = self._pos

    @property
    def _len(self):
        return len(self._input)

    def _lex_start(self):
        raise NotImplementedError

    @property
    def _llen(self):
        return self._pos - self._start

    def _next(self):
        next_char = self._input[self._pos] if self._pos < self._len else None
        self._pos += 1
        return next_char

    @property
    def _peek(self):
        if self._pos >= len(self._input):
            return None
        return self._input[self._pos]

    def _pop_token(self):
        return self._tokens.pop(0)

    def _skip(self, n=1):
        if self._start != self._pos:
            raise ValueError('cannot skip, partial lexeme')
        if self._start +n > len(self._input):
            raise ValueError('cannot skip past EOF')
        self._start = self._pos = self._start + n

    @property
    def _token_in_queue(self):
        while self._tokens:
            token = self._tokens[0]
            if token.symbol is SNTL and self._emit_sntl is False:
                self._pop_token()
                continue
            return True
        return False

class VMLexer(BaseLexer):
    def __init__(self, input, language = None,
                 start_state='_lex_start', emit_sntl=True):


        super(VMLexer, self).__init__(input, start_state, emit_sntl)

    def _lex_start(self):
        assert self._start == self._pos

        peek = self._peek

        if peek is None:
            return self._lex_eof

        elif peek in ' \t\n\r':
            return self._lex_whitespace

        elif peek == '"':
            return self._lex_quoted_string

        elif peek == '#':
            return self._lex_comment

        elif peek in DIGITS:
            return self._lex_number

        else:
            return self._lex_name

    def _lex_eof(self):
        assert self._start == self._pos == self._len
        return None

    def _lex_name(self):
        self._accept_until(WHITESPACE)
        #lexeme = self._input[self._start: self._pos]

        if self._input[self._pos-1] == "'":
            self._emit(STORAGE)
        elif self._input[self._pos-1] == ":":
            self._emit(LABEL)
        else:
            self._emit(NAME)
        return self._lex_start

    def _lex_quoted_string(self):
        self._skip() # over opening quote
        self._accept_until('"')
        self._emit(TEXT)

        # raise unterminated if next character not closing quote
        if self._peek != '"':
            raise SyntaxError("unterminated quote")
        self._skip()

        return self._lex_start

    def _lex_whitespace(self):
        self._accept_run(' \r\n\t')
        self._ignore()
        return self._lex_start

    def _lex_number(self):
        self._accept_run(DIGITS)
        self._emit(NUMBER)
        return self._lex_start


    def _lex_comment(self):
        self._accept_until(EOL)
        self._ignore()
        return self._lex_start



if __name__=='__main__':


    stuff = """ # very simple choice, attack or run if too weak

            mylife 5 - 0 < do_run if
                "fang and claw" attack' . # comment to ignore
                do_run: "run" attack' .
                """

    L = VMLexer(stuff)
    print(L)

    tokens = []
    def go():
        for idx, token in enumerate(L):
            print((idx, token))
            tokens.append(token.lexeme)

    go()
    print(("|".join(tokens), len(tokens)))




