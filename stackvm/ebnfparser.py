# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:00:40 2016

@author: Josh
"""
from string import ascii_lowercase, ascii_uppercase, whitespace
from tokenizer import Symbol, BaseLexer

class TerminalSymbol(Symbol):
    _next_id = 1
    is_terminal = True


class NonTerminalSymbol(Symbol):
    _next_id = 1000
    is_terminal = False


# Tokens for the EBNF Tokenizer and Parser
STARTREPEAT = TerminalSymbol('STARTREPEAT')
ENDREPEAT = TerminalSymbol('ENDREPEAT')
STARTGROUP = TerminalSymbol('STARTGROUP')
ENDGROUP = TerminalSymbol('ENDGROUP')
STARTOPTION = TerminalSymbol('STARTOPTION')
ENDOPTION = TerminalSymbol('ENDOPTION')
OR = TerminalSymbol('OR')
REP = TerminalSymbol('*')
OPT = TerminalSymbol('?')
ATL = TerminalSymbol('+')
SYMBOL = TerminalSymbol('SYMBOL')
LITERAL = TerminalSymbol('LITERAL')
RULE = NonTerminalSymbol('RULE')

class EBNFToken:
    def __init__(self, symbol, lexeme=None):
        self.symbol = symbol
        self.lexeme = lexeme

    def __str__(self):
        return "<EBNFToken {} ({})>".format(self.symbol, self.lexeme)

class EBNFTokenizer(BaseLexer):
    def _emit(self, symbol,):
        BaseLexer._emit(self, symbol)

    def _lex_start(self):
        assert self._start == self._pos

        peek = self._peek

        if peek is None:
            return self._lex_eof

        elif peek in whitespace:
            return self.skip_whitespace()

        elif peek == '"':
            return self.get_quoted_string()

        elif peek in '{}':
            self._skip()
            self._emit(STARTREPEAT if peek=='{' else ENDREPEAT)
            return self._lex_start

        elif peek in '()':
            self._skip()
            self._emit(STARTGROUP if peek=='(' else ENDGROUP)
            return self._lex_start

        elif peek in '[]':
            self._skip()
            self._emit(STARTOPTION if peek=='[' else ENDOPTION)
            return self._lex_start

        elif peek == '|':
            self._skip()
            self._emit(OR)
            return self._lex_start

        elif peek == '*':
            self._skip()
            self._emit(REP)
            return self._lex_start

        elif peek == '?':
            self._skip()
            self._emit(OPT)
            return self._lex_start

        elif peek == '+':
            self._skip()
            self._emit(ATL)
            return self._lex_start

        elif peek in ascii_lowercase:
            return self.get_rule()

        elif peek in ascii_uppercase:
            return self.get_symbol()

    def get_symbol(self):
        self._accept_run(ascii_uppercase)
        self._emit(SYMBOL)
        return self._lex_start

    def get_rule(self):
        self._accept_run(ascii_lowercase)
        self._emit(RULE)
        return self._lex_start


    def skip_whitespace(self):
        self._accept_run(whitespace)
        self._ignore()
        return self._lex_start

    def get_quoted_string(self):
        self._skip() # over opening quote
        self._accept_until('"')
        self._emit(LITERAL)

        # raise unterminated if next character not closing quote
        if self._peek != '"':
            raise SyntaxError("unterminated quote")
        self._skip()

        return self._lex_start

    def _lex_eof(self):
        assert self._start == self._pos == self._len
        return None

class ParserNode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.alternate = False
        self.optional = False
        self.repeating = False
        self.oneormore = False

    def add(self, thing):
        assert isinstance(thing, ParserNode)
        self.children.append(thing)

    def __str__(self):
        if len(self.children) > 1:
            flags = ["alternate" if self.alternate else "",
                     "optional" if self.optional else "",
                     "repeating" if self.repeating else ""]
            if any(flags):
                flagstr = ", ".join(f for f in flags if f)
            else:
                flagstr = "sequence"
        else:
            flagstr = ""
        return "<ParserNode {} ({})>".format(self.name, flagstr)

def make_parser_node(name, tokens):
    if not tokens:
        return None, []
    # print("make_parser_node for {} with {} tokens".format(name, len(tokens)))
    this = ParserNode(name)
    while tokens:
        first = tokens[0]

        if first.symbol in [STARTGROUP, STARTREPEAT, STARTOPTION]:
            # print("found {}".format(first.symbol.name))
            child, tokens = make_parser_node('group', tokens[1:])
            if child:
                this.add(child)

        elif first.symbol in [ENDGROUP, ENDREPEAT, ENDOPTION]:
            # print("Found {} with {} tokens left".format(first.symbol.name, len(tokens)))
            eat = False
            used_brackets = False
            if first.symbol == ENDREPEAT:
                this.repeating = True
                used_brackets = True
            elif first.symbol == ENDOPTION:
                this.optional = True
                used_brackets = True

            if len(tokens) > 1:
                if used_brackets and tokens[1].symbol in [REP, OPT, ATL]:
                    raise SyntaxError("Illegal mix of {} and ()*  in %s" % name)
                if tokens[1].symbol == REP:
                    this.repeating = True
                    eat = True
                elif tokens[1].symbol == OPT:
                    this.optional = True
                    eat = True
                elif tokens[1].symbol == ATL:
                    this.oneormore = True
                    eat = True

            if eat:
                tokens.pop(0)

            return this, tokens[1:]

        elif first.symbol == OR:
            this.alternate = True
            this.add(ParserNode(first))
            tokens.pop(0)

        else:
            this.add(ParserNode(first))
            tokens.pop(0)

    return this, tokens

class EBNFParser(object):
    """
    Class to read EBNF text and return a dictionary of rules
    plus a dictionary of (name, Symbol) pairs
    """

    def __init__(self, text):
        self.symbol_table = {}

        lines = [line for line in text.split(';') if line.strip()]
        data = [line.partition(":=") for line in lines]
        self.rules = {}
        for key, junk, val in data:
            parser_node, remaining = make_parser_node(key.strip(), list(EBNFTokenizer(val)))
            if remaining:
                raise SyntaxError("rule %s did not process correctly" % key)
            self.rules[key.strip()] = parser_node



def print_node(node, indent=0):
    print(" "*indent, str(node))
    for child in node.children:
        print_node(child, indent+2)

if __name__=='__main__':
    text = """expr := term ((PLUS | MINUS) term)*;
            term := factor {(MUL | DIV) factor)};
            factor := NUMBER | "(" expr ")";
            """
    p = EBNFParser(text)
    for key in p.rules:
        print_node(p.rules[key])

