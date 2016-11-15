"""lexer and tokenizer for StackVM
"""

import re

from collections import namedtuple

class Symbol(object):
    """Symbol(name)
    Base class for Terminal and NonTerminal Symbols
    """
    is_terminal = None
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        "The symbol name"
        return self._name

    def __str__(self):
        return "[%s %s]" % (self.__class__.__name__, self.name)

    __repr__ = __str__

class QTerminal(Symbol):
    """Terminal symbol for tokenizers created by :class:`EBNFParser`."""
    is_terminal = True

    
class QNonTerminal(Symbol):
    """Non-terminal symbol for tokenizers created by :class:`EBNFParser`."""
    is_terminal = False

    
class Token(object):
    """Token(symbol, lexeme)
    Container for text (the lexeme) with a symbol defining the role of
    the text.
    """
    __slots__ = ('_symbol', '_lexeme')

    def __init__(self, symbol, lexeme):
        self._symbol = symbol  # always a terminal symbol
        self._lexeme = lexeme

    @property
    def symbol(self):
        """The associated symbol for this token"""
        return self._symbol

    @symbol.setter
    def symbol(self, newsymbol):
        if not isinstance(newsymbol, Symbol):
            raise ValueError("cannot update EBNFToken symbol to non Symbol")
        self._symbol = newsymbol

    @property
    def lexeme(self):
        return self._lexeme

    value = lexeme

    @property
    def is_terminal(self):
        """Returns True if the associated symbol is terminal"""
        return self._symbol.is_terminal


    def __str__(self):
        return "<%s %s: %s >" % (self.symbol.name, self.__class__.__name__, self.lexeme)

    __repr__ = __str__

Lexer = namedtuple('Lexer', 'pattern symbol')


class Tokenizer(object):
    """Tokenizer(text [,token_class])
    Converts text into tokens, based on individual lexers
    """
    def __init__(self, text, token_class=None):
        self.text = text
        self._t_class = token_class or Token
        self.lexers = []
        self._compiled_lexers = {}
        self.symbols = {}

    def __call__(self, text):
        self.text = text
        return self

    def __iter__(self):

        return iter(self.get_next_token, None)

    def add_lexer(self, pattern, symbol_name):
        """add_lexer(patter, symbol_name)
        Adds a regular expression pattern to an internal list for creating
        tokens. If ``symbol_name`` is None, the tokenizer will not emit
        a token when the pattern matches. Otherwise, a :class:`Token` object
        with :class:`QTerminal` symbol will be created for the text that
        matched the pattern.
        """
        if pattern is self._compiled_lexers:
            raise ValueError("lexer for %s already exists" % pattern)

        if symbol_name is None:
            self.lexers.append(Lexer(pattern, None))
        else:
            self.lexers.append(Lexer(pattern,
                                     self.get_symbol(symbol_name)))


    def get_next_token(self):
        """get_next_token()
        Scans the for a token at the beginning of the text.
        Creating the token removes the token's text from the source text.
        Return None if there is no more input.
        """
        if not self.text:
            return None

        for lexer in self.lexers:

            matcher = self.get_matcher(lexer.pattern)
            match = matcher.match(self.text)

            if match:
                lexeme = match.group()
                self.text = self.text[match.end():]
                if lexer.symbol is not None:
                    # print(lexeme, '|'.join(self.text[:15].splitlines()))
                    return self._t_class(lexer.symbol, lexeme)

    def get_symbol(self, name):
        """Returns a symbol for the given name. This method is memoized."""
        return self.symbols.setdefault(name, QTerminal(name))

    def get_matcher(self, pattern):
        """Returns a compiled regex for a pattern. This method is memoized."""
        return self._compiled_lexers.setdefault(pattern, 
                                                re.compile(pattern, re.MULTILINE))


if __name__ == '__main__':
    VMTokenizer = Tokenizer('')
    VMTokenizer.add_lexer('\\s+', None)
    VMTokenizer.add_lexer(r'"[^"]+"', 'LITERAL')
    VMTokenizer.add_lexer(r"[a-zA-Z_]+\:", 'LABEL')
    VMTokenizer.add_lexer(r"[a-zA-Z_]+'", 'STORAGE')
    VMTokenizer.add_lexer(r'#.+$', 'COMMENT')
    VMTokenizer.add_lexer(r'[a-zA-Z_]+', 'NAME')
    VMTokenizer.add_lexer(r'\.', 'STOP')
    VMTokenizer.add_lexer(r'-?[\d.]+', 'NUMBER')
    
    VMTokenizer.add_lexer(r'\S+', 'SYMBOL')
    
    print(list(VMTokenizer(""" # very simple choice, attack or run if too weak
    
            mylife 5 < do_run if
            mylife 10 < do_ttyf if
                "fang and claw" attack' . # comment to ignore
                do_run: "run" attack' .
                do_ttyf: "ttyf" attack' .
                """)))
    
    print(list(VMTokenizer('mylife do_run')))
    print(list(VMTokenizer('mylife do_run \n shananana')))
    print(list(VMTokenizer('mylife do_run # end comment')))
    print(list(VMTokenizer(''' # start comment \n mylife do_run''')))
    print(list(VMTokenizer('99 luftballoons')))
    print(list(VMTokenizer('4 label: item')))
    print(list(VMTokenizer('-9 to go')))
