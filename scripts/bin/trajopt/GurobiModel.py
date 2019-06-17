# external imports
import numpy as np
import gurobipy as grb
from copy import copy, deepcopy
from operator import le, ge, eq


class GurobiModel(grb.Model):
    '''
    Class which inherits from gurobi.Model.
    It facilitates the process of adding (retrieving) multiple variables and
    constraints to (from) the optimization problem.
    '''

    def __init__(self, **kwargs):

        # inherit from gurobi.Model
        super(GurobiModel, self).__init__(**kwargs)

    def add_variables(self, n, lb=None, **kwargs):
        '''
        Adds n optimization variables to the problem.
        It stores the new variables in a numpy array so that they can be readily used for computations.
        Arguments
        ---------
        n : int
            Number of optimization variables to be added to the problem.
        lb : list of floats
            Lower bounds for the optimization variables.
            This is set by default to -inf.
            Note that Gurobi by default would set this to zero.
        Returns
        -------
        x : np.array
            Numpy array that collects the new optimization variables.
        '''

        # change the default lower bound to -inf
        if lb is None:
            lb = [-grb.GRB.INFINITY]*n

        # add variables to the optimization problem
        x = self.addVars(n, lb=lb, **kwargs)

        # update model to make the new variables visible
        # this can inefficient but prevents headaches!
        self.update()

        # organize new variables in a numpy array
        x = np.array(x.values())
        
        return x

    def get_variables(self, name):
        '''
        Gets a set of variables from the problem and returns them in a numpy array.
        Arguments
        ---------
        name : string
            Name of the family of variables we want to get from the problem.
        Returns
        -------
        x : np.array
            Numpy array that collects the asked variables.
        '''

        # initilize vector of variables
        x = np.array([])

        # there cannnot be more x than optimization variables
        for i in range(self.NumVars):

            # get new element and append
            xi = self.getVarByName(name+'[%d]'%i)
            if xi:
                x = np.append(x, xi)

            # if no more elements are available break the for loop
            else:
                break

        return x

    def add_linear_constraints(self, x, operator, y, **kwargs):
        '''
        Adds a linear constraint of the form x (<=, ==, or >=) y to the optimization problem.
        Arguments
        ---------
        x : np.array of floats, gurobi.Var, or gurobi.LinExpr
            Left hand side of the constraint.
        operator : python operator
            Either le (less than or equal to), ge (greater than or equal to), or eq (equal to)
        y : np.array of floats, gurobi.Var, or gurobi.LinExpr
            Right hand side of the constraint.
        Returns
        -------
        c : np.array of gurobi.Constr
            Numpy array that collects the new constraints.
        '''

        # check that the size of the lhs and the rhs match
        assert len(x) == len(y)

        # add linear constraints to the problem
        c = self.addConstrs((operator(x[k], y[k]) for k in range(len(x))), **kwargs)

        # update model to make the new variables visible
        # this can inefficient but prevents headaches!
        self.update()

        # organize the constraints in a numpy array
        c = np.array(c.values())

        return c

    def get_constraints(self, name):
        '''
        Gets a set of constraints from the problem and returns them in a numpy array.
        Arguments
        ---------
        name : string
            Name of the family of constraints we want to get from the problem.
        Returns
        -------
        c : np.array
            Numpy array that collects the asked constraints.
        '''

        # initilize vector of constraints
        c = np.array([])

        # there cannnot be more c than constraints in the problem
        for i in range(self.NumConstrs):

            # get new constraint and append
            ci = self.getConstrByName(name+'[%d]'%i)
            if ci:
                c = np.append(c, ci)

            # if no more constraints are available break the for loop
            else:
                break

        return c

    def add_stage_cost(self, Q, R, x, u, norm):

        # stage cost infinity norm
        if norm == 'inf':

            # add slacks
            sx = self.add_variables(1, lb=[0.])[0]
            su = self.add_variables(1, lb=[0.])[0]
            obj = sx + su

            # enforce infinity norm
            self.add_linear_constraints(Q.dot(x), le, sx)
            self.add_linear_constraints(-Q.dot(x), le, sx)
            self.add_linear_constraints(R.dot(x), le, su)
            self.add_linear_constraints(-R.dot(x), le, su)

        # stage cost one norm
        elif norm == 'one':

            # add slacks
            sx = self.add_variables(x.size, lb=[0.]*x.size)
            su = self.add_variables(u.size, lb=[0.]*u.size)
            obj = sum(sx) + sum(su)

            # enforce one norm
            self.add_linear_constraints(Q.dot(x), le, sx)
            self.add_linear_constraints(-Q.dot(x), le, sx)
            self.add_linear_constraints(R.dot(u), le, su)
            self.add_linear_constraints(-R.dot(u), le, su)

        # stage cost one norm
        elif norm == 'two':
            obj = .5 * (x.dot(Q).dot(x) + u.dot(R).dot(u))

        return obj 

    def add_terminal_cost(self, P, x, norm):

        # terminal cost infinity norm
        if norm == 'inf':

            # add slacks
            s = self.add_variables(1, lb=[0.])[0]
            obj = s

            # enforce infinity norm
            self.add_linear_constraints(P.dot(x), le, np.ones(x.size)*s)
            self.add_linear_constraints(-P.dot(x), le, np.ones(x.size)*s)

        # terminal cost one norm
        elif norm == 'one':

            # add slacks
            s = self.add_variables(x.size, lb[0.]*x.size)
            obj = sum(s)

            # enforce one norm
            self.add_linear_constraints(P.dot(x), le, s)
            self.add_linear_constraints(-P.dot(x), le, s)

        # terminal cost two norm
        elif norm == 'two':
            obj = .5 * x.dot(P).dot(x)

        return obj 