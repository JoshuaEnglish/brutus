# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:00:40 2016

@author: Josh
"""


LIT, SG, EG, OPT, REP, OR, RULE = "LIT SG EG OPT REP OR RULE".split()

class EBNFToken:
    def __init__(self, symbol, lexeme=None):
        self.symbol = symbol
        self.lexeme = lexeme
    
    def __str__(self):
        return "<EBNFToken {} ({})>".format(self.symbol, self.lexeme)
        
class ParserNode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.alternate = False
        self.optional = False
        self.repeating = False

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
        
def make_node(name, tokens):
    if not tokens:
        return None, []
    this = ParserNode(name)
    while tokens:
        first = tokens[0]
        if first.symbol == SG:
            child, tokens = make_node('group', tokens[1:])
            if child:
                this.add(child)
        elif first.symbol == EG:
            eat = False
            if tokens[1].symbol == REP:
                this.repeating = True
                eat = True
            elif tokens[1].symbol == OPT:
                this.optional = True
                eat = True
            
            if eat:
                tokens.pop(0)
            
            return this, tokens[1:] 
            
        elif first.symbol == OR:
            this.alternate = True
            tokens.pop(0)    
  
        else:
            this.add(ParserNode(first))
            tokens.pop(0)
            
    return this, tokens

def print_node(node, indent=0):
    print(" "*indent, str(node))
    for child in node.children:
        print_node(child, indent+2)

if __name__=='__main__':
    node, remainder = make_node('test', [EBNFToken(LIT, "if"),
                                         EBNFToken(SG, ""),
                                         EBNFToken(RULE, 'factor'),
                                         EBNFToken(OR, "or"),
                                         EBNFToken(RULE, 'expr'),
                                         EBNFToken(EG, ""),
                                         EBNFToken(REP, "*")])
    print_node(node)
    print(remainder)