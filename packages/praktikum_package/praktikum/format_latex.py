# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 19:40:18 2023

@author: Balazs
"""
# import time
# t0 = time.perf_counter()

import inspect
import sympy as s
from math import sqrt
import matplotlib.pyplot as plt
# t1 = time.perf_counter()
# print('imports:', t1-t0)

class Var:
    def __init__(self, string):
        self.value = string
    
    def __str__(self):
        return self.value
    def __repr__(self):
        return 'v:'+str(self.value)

class Sym:
    def __init__(self, string):
        if string in ['+', '-', '*', '/', '**', '^', '\\']:
            self.value = string
        else:
            raise ValueError(f'"{string}" is not a symbol')
    
    def __str__(self):
        return self.value
    def __repr__(self):
        return 's:'+str(self.value)

class Num:
    def __init__(self, string):
        try:
            self.value = string
            self.int = float(string)
        except ValueError:
            raise ValueError(f'"{string}" is not a number')
    
    def __str__(self):
        return self.value
    def __repr__(self):
        return 'n:'+str(self.value)
    
class Bra:
    def __init__(self, string):
        if string in ['(', '[', '{']:
            self.value = string
            self.int = 1
        elif string in [')', ']', '}']:
            self.value = string
            self.int = -1
        else:
            raise ValueError(f'"{string}" is not a bracket')
    
    def __str__(self):
        return self.value
    def __repr__(self):
        return 'b:'+str(self.value)

class Fun:
    def __init__(self, string):
        self.value = string
    
    def __str__(self):
        return self.value
    def __repr__(self):
        return 'f:'+str(self.value)

def tokenize(expression):
    string = ''
    tokens = []
    i = 0
    while i < len(expression):
        if expression[i] == ' ':
            pass
        elif expression[i].isalpha() or expression[i] == '\\':
            while i < len(expression) and (expression[i].isalnum() or expression[i] in ['.', '_', '\\']):
                string += expression[i]
                i += 1
            i -= 1
            tokens.append(Var(string))
            string = ''
            
        elif expression[i].isnumeric():
            while i < len(expression) and (expression[i].isnumeric() or expression[i] == '.'):
                string += expression[i]
                i += 1
            i -= 1
            tokens.append(Num(string))
            string = ''
        
        elif i < len(expression) and expression[i] in ['(', ')', '[', ']', '{', '}']:
            tokens.append(Bra(expression[i]))
        
        else:
            while i < len(expression) and not expression[i].isalnum() and not expression[i] in ['(', ')', '[', ']', '{', '}', ' ']:
                string += expression[i]
                i += 1
            i -= 1
            tokens.append(Sym(string))
            string = ''
        i += 1
    
    for i, token in enumerate(tokens):
        if i == len(tokens)-1:
            break
        if isinstance(token, Var) and isinstance(tokens[i+1], Bra):
            tokens[i] = Fun(token.value)
    return tokens

def extract_f(f, parms = None):
    string = inspect.getsourcelines(f)[0]
    name = string[0][4:string[0].find('(')]
    name0 = name[:]
    
    if '_' in name:
        name = name[:name.find('_')] + '_{' + name[name.find('_')+1:] + '}'
        name0 = name0[:name0.find('_')] + '-' + name0[name0.find('_')+1:]
    
    if len(string) == 2:
        expression = string[-1][11:].strip()
    else:
        print("WARNING!!! The formula (function) can't be read")
        return ''
    
    parms0 = string[0][string[0].find('(')+1:-3].split(',')
    for i in range(len(parms0)):
        parms0[i] = parms0[i].strip()
    if parms != None:
        tokens = tokenize(expression)
        for i in range(len(tokens)):
            for parm0, parm in zip(parms0, parms):
                if tokens[i].value == parm0:
                    tokens[i].value = parm
        expression = ''
        for token in tokens:
            expression += token.value
        parms0 = parms
    expression = expression.replace('np.log', 'ln')
    expression = expression.replace('np.log10', 'log')
    expression = expression.replace('np.', '')
    
    return name0, name, parms0, expression

def format_to_latex(f, parms = None):
    '''
    Parameters
    ----------
    f : formula defined in two lines using only elementary functions,
    works only if this function recognises it

    Returns
    -------
    a string formatted to latex containing the formula

    '''
    
    # name0, name, parms0, expression = extract_f(f, parms)
    name0, name, parms0, expression = extract_f(f)
    # print(s.parsing.sympy_parser.parse_expr(expression))
    expr_latex = s.latex(s.parsing.sympy_parser.parse_expr(expression))
    # if parms:
    #     tokens = tokenize(expr_latex)
    #     print(tokens)
    #     for parm0, parm in zip(parms0, parms):
    #         for tok in tokens:
    #             if isinstance(tok, Var) and tok.value == parm0:
    #                 tok.value = parm
               
    #     expr_latex = ''.join([tok.value for tok in tokens])
    
    return '\\begin{equation}\n\t' + "\\label{" + name0 + "}\n\t" + name + ' = ' + expr_latex + '\n\\end{equation}'

def derivate(f_expression, dp):
    '''
    

    Parameters
    ----------
    f_expression : expression in form of a string or sympy expression
    dp : string,
         variable according to which the function will de derivated.

    Returns
    -------
    A string containing the derivative of the function 

    '''
    return str(s.diff(f_expression, dp))
    

from threading import Thread
import time
def expr(chyba, ret):
    ret.append(s.latex(s.simplify(chyba)))
    
def prenos_chyb_latex(f, parms = None):
    '''
    
    Parameters
    ----------
    f : function
        defined on 2 lines using only elementary functions.
    parms : iterable, optional
        containing the names of the parametres of the function, 
        if None, the original names are used. The default is None.

    Returns
    -------
    string
        formatted to the style of latex.

    '''
    
    chyba = ''
    der = ''
    name0, name, parms0, expression = extract_f(f, parms)
    
    for i, p in enumerate(parms0):
        der = str(derivate(expression, p))
        chyba += f'({str(s.simplify(der+"*"+der))})*sigma_{p}**2 +'
        #print(str(s.simplify(der+"*"+der)))
    chyba = chyba[:-1]
    try:
        ret = []
        t1 = Thread(target=expr, args=(chyba, ret))
        t1.start()
        t0 = time.perf_counter()
        time.sleep(0.001)
        while t1.is_alive() and time.perf_counter()-t0 < 1:
            time.sleep(0.001)
        if t1.is_alive():
            raise TimeoutError()
        t1.join()
        expr_latex = ret[0]
    except:
        expr_latex = s.latex(s.parsing.sympy_parser.parse_expr(chyba))
    
    return '\\begin{equation}\n\t' + "\\label{sigma-" + name0 + "}\n\t\\sigma_{" + name + '}^2 = ' + expr_latex + '\n\\end{equation}'

def prenos_chyb_eval(f, errs, parms):
    '''
    Parameters
    ----------
    f : function
        defined on 2 lines using only elementary functions.
    parms : iterable, optional
        containing the names of the parametres of the function, 
        if None, the original names are used. The default is None.

    Returns
    -------
    the standard devation of the value that is defined by the function.

    '''
    
    chyba = ''
    der = ''
    name0, name, parms0, expression = extract_f(f)
    
    for i, p in enumerate(parms0):
        der = str(derivate(expression, p))
        chyba += f'({str(s.simplify(der+"*"+der))})*sigma_{p}**2 +'
        #print(str(s.simplify(der+"*"+der)))
    chyba = chyba[:-1]
    
    tokens = tokenize(chyba)
    
    parms_errs_dict = {}
    for i, parm in enumerate(parms0):
        parms_errs_dict[parm] = parms[i]
        parms_errs_dict[f'sigma_{parm}'] = errs[i]
    
    for key, val in parms_errs_dict.items():
        for i in range(len(tokens)):
            if tokens[i].value == key:
                tokens[i].value = str(parms_errs_dict[key])
    
    expression = ''
    for token in tokens:
        expression += token.value
    
    err = s.parsing.sympy_parser.parse_expr(expression).evalf()
    return sqrt(err)

def default_table(df, label, caption, header = None):
    nc = len(df.columns)+1
    form = '|'+'|'.join(['l']*nc)+'|'
    cline = '\t\t\t\\cline{1-' + str(nc-1) + '}'
    df = df.style.to_latex(column_format = form,
                           label = label,
                           caption = caption)
    df = df.split('\n')
    for i in range(len(df)):
        if i>0 and i<3:
            df[i] = '\t' + df[i]
        elif i >= 3 and i < len(df)-2:
            df[i] = '\t\t' +df[i]
        else:
            df[i] = '\t' + df[i]
    
    dfn = ['\\begin{table}[H]']
    dfn.extend(df[1:3])
    dfn.append('\t\\begin{center}')
    dfn.append('\t\\begin{tabular}{' + form + '}')
    dfn.append(cline)
    
    if header != None:
        dfn.append(df[3])
        dfn.append(cline)
        head_line = ''
        for col, head in header:
            head_line += f'&\\multicolumn{"{"}{col}{"}{|l|}{"}{head}{"}"} '
        head_line += '\\\\'
        dfn.append(head_line[1:])
        dfn.append(cline)
    
        for i in range(4, len(df)-3):
            ln = df[i]
            ln = ln.split('&')[1:]
            dfn.append('&'.join(ln))
            dfn.append(cline)
    else:
        for i in range(4, len(df)-3):
            ln = df[i]
            ln = ln.split('&')[1:]
            dfn.append('&'.join(ln))
            dfn.append(cline)
        
    dfn.append('\t\\end{tabular}')
    dfn.append('\t\\end{center}')
    dfn.append('\\end{table}')
    return '\n'.join(dfn)


def display(latex_equation, dpi = 120):
    '''
    Intended use:
        display(prenos_chyb_latex(f, parms))
    It displays a latex equation which has a formatting:
        \begin{equation}
        \label{...}
        ...
        \end{equation}

    Parameters
    ----------
    latex_equation : str
        The latex equation to display formatted as above.

    Returns
    -------
    None.

    '''
    
    fig = plt.figure(dpi = dpi)
    ax = fig.add_subplot()
    
    text = ax.text(0.5, 0.5, '$' + ''.join(latex_equation.split()[2:-1]) + '$', fontsize=16, ha='center', va='center')
    
    # Step 2: Remove axes for clean display
    ax.axis('off')
    
    # Step 3: Render the figure to get the text bounding box
    fig.canvas.draw()
    bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
    
    # Step 4: Convert bounding box from display to figure coordinates
    bbox_in_fig_coords = bbox.transformed(fig.transFigure.inverted())
    width, height = bbox_in_fig_coords.width, bbox_in_fig_coords.height
    
    # Step 5: Set figure size based on bounding box
    fig.set_size_inches(width+1, height+0.8)
    
    fig_manager = plt.get_current_fig_manager()
    fig_manager.resize(int(width * dpi), int((height+0.8) * dpi))
    
    fig.canvas.draw()
    plt.show()
    
    # plt.figure(dpi = dpi)
    # plt.axis('off')
    # plt.text(0.5, 0.5, '$' + ''.join(latex_equation.split()[2:-1]) + '$', ha='center', va='center')

# print('definitions:', time.perf_counter()-t1)