# -*- coding: utf-8 -*-
"""
Finite State Machine parsing

Attempt to use state machine patterns to do this cleanly

http://www.python-course.eu/finite_state_machine.php
    I found this too limiting. Can only do one token lookahead
    
https://python-3-patterns-idioms-test.readthedocs.io/en/latest/StateMachine.html
    Also only one token lookahead
    
I think what I need is something like my old L-system generator.
Reviewing that is not as helpful as I'd like, but I think I know how to try this.

"""
#from ebnf import CSTNode, EBNFNode, EBNFParser
class CSTNode(object):
    """Concrete Syntax Tree Node. 
    This node holds a token and a list of children for that token.
    """
    def __init__(self, token):
        self.token = token
        self.children = []

    def __str__(self):
        return "<CSTNode:{} >".format(self.token)
        

class State:
    def __init__(self, parsernode, tokens):
        self.parsernode = parsernode
        self.tokens = tokens
        self.cstnode = None
    
    def make_cstnode(self):
        raise NotImplementedError
        
class TerminalState(State):
    def make_cstnode(self):
        """create a cstnode if possible. Return True if successful, False otherwise"""
        
        
class StateMachine:
    def __init__(self):
        self.handlers = {}
        self.startState = None
        self.endStates = []

    def add_state(self, name, handler, end_state=0):
        name = name.upper()
        self.handlers[name] = handler
        if end_state:
            self.endStates.append(name)
    
    def set_start(self, name):
        self.startState = name.upper()
        
    def run(self, cargo):
        try:
            handler = self.handlers[self.startState]
        except:
            raise ValueError("must call .set_start() before .run()")
        
        if not self.endStates:
            raise ValueError("at least one state must be an end_state")
        
        while True:
            (newState, cargo) = handler(cargo)
            if newState.upper() in self.endStates:
                print("reached", newState)
                break
            else:
                handler = self.handlers[newState.upper()]

positive_adjectives = ['great', 'super', 'fun', 'entertaining', 'easy']
negative_adjectives = ['boring', 'difficult', 'ugly', 'bad']

def start_transitions(txt):
    splitted_txt = txt.split(None,1)
    word, txt = splitted_txt if len(splitted_txt) > 1 else (txt,"")
    if word == "Python":
        newState = "Python_state"
    else:
        newState = "error_state"
    return (newState, txt)

def python_state_transitions(txt):
    splitted_txt = txt.split(None,1)
    word, txt = splitted_txt if len(splitted_txt) > 1 else (txt,"")
    if word == "is":
        newState = "is_state"
    else:
        newState = "error_state"
    return (newState, txt)

def is_state_transitions(txt):
    splitted_txt = txt.split(None,1)
    word, txt = splitted_txt if len(splitted_txt) > 1 else (txt,"")
    if word == "not":
        newState = "not_state"
    elif word in positive_adjectives:
        newState = "pos_state"
    elif word in negative_adjectives:
        newState = "neg_state"
    else:
        newState = "error_state"
    return (newState, txt)

def not_state_transitions(txt):
    splitted_txt = txt.split(None,1)
    word, txt = splitted_txt if len(splitted_txt) > 1 else (txt,"")
    if word in positive_adjectives:
        newState = "neg_state"
    elif word in negative_adjectives:
        newState = "pos_state"
    else:
        newState = "error_state"
    return (newState, txt)

def neg_state(txt):
    print("Hallo")
    return ("neg_state", "")

if __name__== "__main__":
    m = StateMachine()
    m.add_state("Start", start_transitions)
    m.add_state("Python_state", python_state_transitions)
    m.add_state("is_state", is_state_transitions)
    m.add_state("not_state", not_state_transitions)
    m.add_state("neg_state", None, end_state=1)
    m.add_state("pos_state", None, end_state=1)
    m.add_state("error_state", None, end_state=1)
    m.set_start("Start")
    m.run("Python is great")
    m.run("Python is difficult")
    m.run("Perl is ugly")
        