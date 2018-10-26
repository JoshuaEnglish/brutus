# -*- coding: utf-8 -*-
"""
Sample multi-line 4-function calculator.

Results from previous lines can be used in later lines.

The order of operations is implied in how the REBNF is defined.
"""
from brutus import Parser, Coder, BaseMachine
from brutus.utils import print_xml, print_node

text = """statements := assignment { assignment } ;
        assignment := VAR STORE expr STOP [COMMENT];
        expr := term { EXPROP term };
        term := factor { TERMOP factor };
        factor := INTEGER | VAR | "(" expr ")";
        VAR := [a-z]+;
        INTEGER := -?[0-9]+;
        STORE := <-;
        EXPROP := [-+];
        TERMOP := [*/];
        STOP := [\.];
        PARENS := [()];
        COMMENT := #.*$;
        """

p = Parser(text)
print("Parser Tree for statements:")
print_node(p.rules['statements'])

print("\nParser Tree for assignment:")
print_node(p.rules['assignment'])

print("\nParser Tree for factor")
print_node(p.rules['factor'])

double = """a <- 2*7+6/3 . # we're gonna use this in a second
b<-a/2."""
simple = """x <- 2 - 1. # some comment"""

program = double
print("\nProgram:")
print(program)

print("\nTokens:")
print(list(p.tokenizer(program)))
try:
    ok, node, detritus = p.parse_text(program)
    print("\nConcrete Syntax Tree:")
    print_xml(node)
    print(detritus)
except SyntaxError as E:
    print("\nERROR LOG")
    p.report()
    raise E


class MathCoder(Coder):
    encode_integer = Coder.encode_terminal

    encode_exprop = Coder.handle_terminal
    encode_termop = Coder.handle_terminal
    encode_parens = Coder.do_nothing
    encode_var = Coder.handle_terminal
    encode_factor = Coder.handle_children
    encode_statements = Coder.handle_children
    encode_expr = Coder.handle_binary_node
    encode_term = Coder.handle_binary_node
    encode_comment = Coder.do_nothing

    def encode_assignment(self, node):
        variable, op, stuff, stop = node.children[:4]
        self.handle_node(stuff)
        self.code.append("{}'".format(variable.token.lexeme))


coder = MathCoder()
code = coder.encode(node)
print("Code:", code)


mather = BaseMachine('math')
mather.feed(code)
mather.run()
print(mather.registers)
