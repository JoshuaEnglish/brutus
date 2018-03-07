import unittest

from brutus import Parser, Coder, BaseMachine


class IfThenCoder(Coder):
    encode_integer = Coder.encode_terminal
    encode_op = Coder.handle_terminal
    encode_parens = Coder.do_nothing

    encode_name = Coder.handle_terminal
    encode_literal = Coder.handle_terminal
    encode_factor = Coder.handle_children
    encode_statements = Coder.handle_children
    encode_expr = Coder.handle_binary_node
    encode_term = Coder.handle_binary_node

    def encode_assignment(self, node):
        variable, op, stuff = node.children
        self.handle_node(stuff)
        self.code.append("{}'".format(variable.token.lexeme))

    def encode_ifstmt(self, node):
        # target comp if
        # target else comp ife
        if 'else' not in (t.token.lexeme for t in node.children):
            # we've got a regular if statement
            # cond not endif if if_true endif:

            endif = self.make_label('if')

            self.handle_node(node.children[2])  # name
            self.code.append('not')
            self.code.append(endif)

            self.code.append('if')

            self.handle_node(node.children[5])  # if true clause
            self.code.append('{}:'.format(endif))
        else:
            iftrue = self.make_label('if')
            iffalse = self.make_label('if')
            endif = self.make_label('if')
            self.handle_node(node.children[2])  # name
            self.code.append(iftrue)
            self.code.append(iffalse)

            self.code.append('ife')

            self.code.append('{}:'.format(iftrue))
            self.handle_node(node.children[5])
            self.code.append(endif)
            self.code.append('jump')
            self.code.append('{}:'.format(iffalse))
            self.handle_node(node.children[9])
            self.code.append('{}:'.format(endif))


class IfThenExample(unittest.TestCase):
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
    coder = IfThenCoder()

    def test_ifonly(self):
        ok, node, detritus = self.p.parse_text('if (a) (res <- "marmalade")')
        self.assertTrue(ok)
        self.assertListEqual(detritus, [])
        code = self.coder.encode(node)
        chooser = BaseMachine('ifonly')
        chooser.feed(code)
        chooser.run(a=1, res="blankenship")
        self.assertEqual(chooser.registers['res'], '"marmalade"')
        chooser.run(a=0, res="blankenship")
        self.assertEqual(chooser.registers['res'], '"blankenship"')

    def test_ifelse(self):
        ok, node, detritus = self.p.parse_text(
            """if (a) (res <- "marmalade") else (res <- "blankenship")""")
        self.assertTrue(ok)
        self.assertListEqual(detritus, [])
        code = self.coder.encode(node)
        chooser = BaseMachine('ifthen')
        chooser.feed(code)
        chooser.run(a=1, res="python")
        self.assertEqual(chooser.registers['res'], '"marmalade"')
        chooser.run(a=0, res="python")
        self.assertEqual(chooser.registers['res'], '"blankenship"')


if __name__ == '__main__':
    unittest.main()
