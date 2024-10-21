import torch
from bnn import BinaryBlock
import math
from typing import Tuple, List
from pysat.formula import (
    CNF, Equals, Formula, Atom, IDPool, And, Or, Neg, PYSAT_TRUE, PYSAT_FALSE
)
from pysat.card import CardEnc
from pysat.solvers import Glucose3


def get_sequential_counter(
        ip_variables: List[int], bound: int
    ) -> Formula:

    # cnf = CardEnc.atleast(
    #     lits=Formula.literals(ip_variables), bound=bound, #vpool=id_pool, encoding=1
    # )

    # return CNF(from_clauses=cnf.clauses)

    counter = 0
    for x_i, w in zip(x, Formula.literals(ip_variables)):
        if ((x_i > 0) and (w>0)) or (((x_i < 0) and (w<0))):
            counter += 1
    return PYSAT_TRUE if counter >= bound else PYSAT_FALSE


def equals(cnf, y_i):

    if cnf == PYSAT_TRUE:
        return CNF(from_clauses=[Formula.literals([y_i])])
    else:
        return CNF(from_clauses=[Formula.literals([Neg(y_i)])])
        


def get_binary_block_cnf(
        f: BinaryBlock, ip_variables: List[int], op_variables: List[int], solver: Glucose3
    ):

    D = (
        -f.linear.bias/2 + f.linear.weight.sum(dim=1)/2 + (f.linear.weight < 0).sum(dim=1)
    ).tolist()

    weight = f.linear.weight.tolist()

    for d_i, y_i, w_i in zip(D, op_variables, weight):

        bound = int(math.ceil(d_i))
        if bound > len(ip_variables):
            solver.append_formula(CNF(from_clauses=[Formula.literals([Neg(y_i)])]))
        elif bound <= 0:
            solver.append_formula(CNF(from_clauses=[Formula.literals([y_i])]))
        else:

            ######################################################
            # sum(x_b with G_m > 0) + sum(x_b with G_m < 0)
            new_ip_vars = [ip_var if w>0 else Neg(ip_var) for ip_var, w in zip(ip_variables, w_i)]
            
            ######################################################
            cnf = get_sequential_counter(new_ip_vars, bound=bound)
            # If the literals satisfy the condition then output is 1 else it is 0
            cnf = equals(cnf, y_i)
            ######################################################
            
            solver.append_formula(cnf)


def get_op(f: BinaryBlock, x: List[int]) -> List[int]:
    return f(torch.tensor([x])).tolist()

def get_op_from_solver(f: BinaryBlock, x: List[int]) -> List[int]:

    g = Glucose3()

    ip_variables = [Atom(f'I{i}') for i in range(f.in_features)]
    op_variables = [Atom(f'O{i}') for i in range(f.out_features)]

    get_binary_block_cnf(f, ip_variables, op_variables, g)

    assumptions = [Formula.literals([ip])[0] if x > 0 else Formula.literals([Neg(ip)])[0] 
               for ip, x in zip(ip_variables, x)]
    
    g.solve(assumptions=assumptions)
    model = g.get_model()

    # Translating the output from True/False to 1/-1
    x_output = []
    for x_op in Formula.literals(op_variables):
        for op_var in model:
            if op_var == x_op:
                x_output.append(1.0)
            elif op_var == - x_op:
                x_output.append(-1.0)
    return [x_output]


x = None
if __name__ == '__main__':
    f = BinaryBlock(3, 5).eval()
    x = [1.0, 1.0, -1.0]


    print(get_op(f, x))
    print(get_op_from_solver(f, x))


