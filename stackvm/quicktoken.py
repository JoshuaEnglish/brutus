# -*- coding: utf-8 -*-
"""
quicktokenizer

Idea to build a tokenizer based on generated rules
Created on Sat Nov  5 18:08:31 2016

@author: Josh
"""
import re
from collections import namedtuple, OrderedDict

class Symbol(object):
    is_terminal = None
    def __init__(self, name):
        self._name = name
    
    @property
    def name(self):
        return self._name
    
    def __str__(self):
        return "[%s %s]" % (self.__class__.__name__, self.name)
    
    __repr__ = __str__

class QTerminal(Symbol):
    is_terminal = True

class QNonTerminal(Symbol):
    is_terminal = False
    
class Token(object):
    __slots__ = ('_symbol', '_lexeme')

    def __init__(self, symbol, lexeme):
        self._symbol = symbol  # always a terminal symbol
        self._lexeme = lexeme

    @property
    def symbol(self):
        return self._symbol
    
    @property
    def lexeme(self):
        return self._lexeme
    
    value = lexeme

    def __str__(self):
        return "<%s %s: %s >" % (self.symbol.name, self.__class__.__name__, self.lexeme)
        
    __repr__ = __str__
    
Lexer = namedtuple('Lexer', 'pattern symbol')

class Tokenizer(object):
    def __init__(self, text, token_class = None):
        self.text = text
        self._t_class = token_class or Token
        self.lexers = []
        self._compiled_lexers = {}
        self.symbols = {}
        self._token_count = 0

    def __iter__(self):
        
        return iter(self.get_next_token, None)
    
    def add_lexer(self, pattern, symbol_name):
        if pattern is self._compiled_lexers:
            raise ValueError("lexer for %s already exists" % pattern)
        
        if symbol_name is None:
            self.lexers.append(Lexer(pattern, None))
        else: 
            self.lexers.append(Lexer(pattern, 
                                     self.get_symbol(symbol_name)))
  
    
    def get_next_token(self):
        if not self.text:
            return None
        if self._token_count == 10:
            pass
        self._token_count += 1
        #print("Text:", self.text)
        for lexer in self.lexers:
            
            matcher = self.get_matcher(lexer.pattern)
            match = matcher.match(self.text)
            #print("trying %s, got %s" % (lexer.pattern, match))
            
            if match:
                lexeme = match.group()
                #print("found %s, emiting %s" % (lexeme, lexer.symbol))
                self.text = self.text[match.end():]
                if lexer.symbol is not None:
                    return self._t_class(lexer.symbol, lexeme)
                
    def get_symbol(self, name):
        return self.symbols.setdefault(name, QTerminal(name))
            
    def get_matcher(self, pattern):
        return self._compiled_lexers.setdefault(pattern, re.compile(pattern))



def test_ebnf_parser():
    text = """statements := assignment { assignment } ;
            assignment := var "<-" expr ".";
            expr := term {("+" | "-") term};
            term := factor {("*" | "/") factor};
            factor := integer | var | "(" expr ")";
            var := "a" | "b" | "c";
            integer := ["+" | "-"] digit {digit};
            digit := "0" | "1" | "2" | "3" | "4" |
                     "5" | "6" | "7" | "8" | "9" ;
            """

    lines = [line for line in text.split(';') if line.strip()]
    data = [line.split(":=") for line in lines]
    rules = OrderedDict()
        
    E = Tokenizer('')
    E.add_lexer('\s+', None)
    E.add_lexer('[a-z]+', 'RULE')
    E.add_lexer('[A-Z]+', 'TERM')
    
    E.add_lexer(r'"[^"]+"', 'LITERAL')
    
    E.add_lexer("[{]", 'STARTREPEAT')
    E.add_lexer("[}]", 'ENDREPEAT')
    E.add_lexer("[(]", 'STARTGROUP')
    E.add_lexer("[)]", 'ENDGROUP')
    E.add_lexer("[[]", 'STARTOPTIONAL')
    E.add_lexer("[]]", 'ENDOPTIONAL')
    E.add_lexer("[|]", 'OR')
    E.add_lexer(":=", 'DEFINE')
    E.add_lexer(";", 'ENDDEFINE')
    for key, definition in data:
        E.text = definition
        rules[key.strip()] = list(E)
        if E.text:
            print("line not consumed:", E.text)

if __name__ == '__main__':
    from ebnf import EBNFParser
    from coder import Coder
    text = """statements := assignment { assignment } ;
            assignment := VAR "<-" expr ".";
            expr := term {("+" | "-") term};
            term := factor {("*" | "/") factor};
            factor := INTEGER | VAR | "(" expr ")";
            var := "a" | "b" | "c";
            integer := ["+" | "-"] digit {digit};
            digit := "0" | "1" | "2" | "3" | "4" |
                     "5" | "6" | "7" | "8" | "9" ;
            """
    p = EBNFParser(text)
    ### this is a tokenizer for the language defined by the EBNF
    # should accept "a <- 12 * 7 + 3 * 2 . b <- a / 2 . res <- b * d ."
    
    T = Tokenizer("a <- 12*7+3*2 . \n\tb<-a/2. res <- b * d .")

    T.add_lexer('\s+', None)
    T.add_lexer('[a-z]+', 'VAR')
    T.add_lexer('[0-9]+', 'INTEGER')
    T.add_lexer('[+\-*/]', 'OP')
    T.add_lexer('[()]', 'PARENS')
    T.add_lexer('[.]', 'STOP')
    T.add_lexer('<-', 'STORE')
    

    node, detritus = p.parse(list(T))
        
    class MathCoder(Coder):
        encode_integer = Coder.encode_terminal
        encode_op = Coder.handle_terminal
        encode_parens =Coder.do_nothing
        encode_var = Coder.handle_terminal
        encode_factor = Coder.handle_children
        encode_statements = Coder.handle_children
        encode_expr = Coder.handle_binary_node
        encode_term = Coder.handle_binary_node

        def encode_assignment(self, node):
            #print("assignment", node, token_lexemes(node.children))
            variable, op, stuff, stop = node.children
            self.handle_node(stuff)
            self.code.append("{}'".format(variable.token.lexeme))
    
    coder = MathCoder()
    code = coder.encode(node)
    print("Code:", code)
    
    from machine import BaseMachine
    mather = BaseMachine('math')
    mather.feed(code)
    mather.run(d=1)
    print(mather.registers)