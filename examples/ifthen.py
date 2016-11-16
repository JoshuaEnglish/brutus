# -*- coding: utf-8 -*-
"""
Sample if - then statement
"""
from stackvm import Parser, Coder, BaseMachine
from stackvm.utils import print_xml

text = """ifstmt := "if" "(" NAME ")" "(" assignment ")" {"else" "(" assignment ")"} ;
        assignment := NAME STORE expr;
        expr := term {("+" | "-") term};
        term := factor {("*" | "/") factor};
        factor := INTEGER | NAME | LITERAL | "(" expr ")";
        NAME := [a-z]+;
        INTEGER := -?[0-9]+;
        STORE := <-;
        OP := [+\-*/];
        PARENS := [()];
        """

p = Parser(text)
program = """if (a) (res <- "true")
else (res <- "false" ) """
print(list(p.tokenizer(program)))
node, detritus = p.parse_text(program)
print_xml(node)
print(detritus)

#class MathCoder(Coder):
#    encode_integer = Coder.encode_terminal
#    encode_op = Coder.handle_terminal
#
#    encode_mul = encode_div = encode_plus = encode_minus = Coder.handle_terminal
#    encode_parens = Coder.do_nothing
#    encode_op = encode_cp = Coder.do_nothing
#    encode_var = Coder.handle_terminal
#    encode_factor = Coder.handle_children
#    encode_statements = Coder.handle_children
#    encode_expr = Coder.handle_binary_node
#    encode_term = Coder.handle_binary_node
#
#    def encode_assignment(self, node):
#        #print("assignment", node, token_lexemes(node.children))
#        variable, op, stuff = node.children
#        self.handle_node(stuff)
#        self.code.append("{}'".format(variable.token.lexeme))
#
#coder = MathCoder()
#code = coder.encode(node)
#print("Code:", code)
#
#
#mather = BaseMachine('math')
#mather.feed(code)
#mather.run()
#print(mather.registers)