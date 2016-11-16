# -*- coding: utf-8 -*-
"""
Coder

Translates :class:`CSTNodes` to code for the stack machine
"""
from collections import Counter


class Coder(object):
    """code generation tool"""

    def __init__(self):
        self.code = []
        self._labels = Counter()

    def make_label(self, name):
        """returns a new label in a sequence"""
        this = self._labels[name]
        label = "{}_{}".format(name, this)
        self._labels[name] += 1
        return label

    def encode(self, node):
        """return a space deliniated string of items of code.
        Automatically appends 'END' if needed.
        """
        self.handle_node(node)
        if self.code[-1] not in ['.', 'END', 'end']:
            self.code.append('END')
        return " ".join(self.code)

    def handle_terminal(self, node):
        """adds the token's lexeme to the code output"""
        self.code.append(node.token.lexeme)

    def encode_terminal(self, node):
        """adds the token's lexeme to the code output"""
        self.code.append(node.token.lexeme)

    def handle_binary_node(self, node):
        """converts a repeating infix operation to postfix in the code output"""
        self.handle_node(node.children[0])
        idx = 1
        while idx < len(node.children):
            self.handle_node(node.children[idx+1])
            self.handle_node(node.children[idx])
            idx += 2


    def handle_node(self, node):
        """determines which encode_xxx function to call based on the CSTNode's
        token or symbol name.
        """
        func_name = "encode_{}".format(node.token.lexeme)
        symbol_name = "encode_{}".format(node.token.symbol.name.lower())
        if hasattr(self, func_name):
            getattr(self, func_name)(node)
        elif hasattr(self, symbol_name):
            getattr(self, symbol_name)(node)
        else:
            raise SyntaxError("Coder cannot handle %s" % node)

    def handle_children(self, node):
        """encodes children, ignoring the parent node"""
        for child in node.children:
            self.handle_node(child)

    def do_nothing(self, node):
        """ignores the node in code generation"""
        pass