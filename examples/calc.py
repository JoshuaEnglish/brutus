# -*- coding: utf-8 -*-
"""
Sample multi-line 4-function calculator.

Results from previous lines can be used in later lines.
"""
from stackvm import Parser, Coder, BaseMachine
from stackvm.utils import print_xml

text = """statements := assignment { assignment } ;
        assignment := VAR STORE expr STOP;
        expr := term {("+" | "-") term};
        term := factor {("*" | "/") factor};
        factor := INTEGER | VAR | "(" expr ")";
        VAR := [a-z]+;
        INTEGER := -?[0-9]+;
        STORE := <-;
        BINOP := [+\-*/];
        STOP := [\.];
        PARENS := [()];
        
        """

p = Parser(text)
program = """a <- 2*(7+3)*4 . 
b<-a/2."""
print(list(p.tokenizer(program)))
node, detritus = p.parse_text(program)
print_xml(node)
print(detritus)

class MathCoder(Coder):
    encode_integer = Coder.encode_terminal
    
    encode_binop = Coder.handle_terminal
    encode_parens = Coder.do_nothing
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


mather = BaseMachine('math')
mather.feed(code)
mather.run()
print(mather.registers)