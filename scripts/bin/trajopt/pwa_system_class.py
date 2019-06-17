import numpy as np


class AffineSystem(object):
    """
    Discrete-time affine systems in the form x(t+1) = A x(t) + B u(t) + c.
    """

    def __init__(self, A, B, c):
        """
        Initializes the discrete-time affine system.
        Arguments
        ----------
        A : numpy.ndarray, dimension nx by nx
            State transition matrix (assumed to be invertible).
        B : numpy.ndarray, dimension nx by nu
            Input to state map.
        c : numpy.ndarray, dimension nx
            Offset term in the dynamics.
        """
        # store inputs
        self.A = A
        self.B = B
        self.c = c

        # system size
        self.nx, self.nu = B.shape

class Domain:
	def __init__(self, A, b):
		self.A = A
		self.b = b


class PiecewiseAffineSystem(object):
    """
    Discrete-time piecewise-affine systems in the form x(t+1) = A_i x(t) + B_i u(t) + c_i if (x(t), u(t)) in D_i := {(x,u) | F_i x + G_i u <= h_i}.
    """
    def __init__(self, affine_systems, domains):
        # same number of systems and domains
        if len(affine_systems) != len(domains):
            raise ValueError('the number of affine systems has to be equal to the number of domains.')

        # same number of states for each system
        nx = set(S.nx for S in affine_systems)
        if len(nx) != 1:
            raise ValueError('all the affine systems must have the same number of states.')
        self.nx = list(nx)[0]

        # same number of inputs for each system
        nu = set(S.nu for S in affine_systems)
        if len(nu) != 1:
            raise ValueError('all the affine systems must have the same number of inputs.')
        self.nu = list(nu)[0]

        # same dimensions for each domain
        nxu = set(D.A.shape[1] for D in domains)
        if len(nxu) != 1:
            raise ValueError('all the domains must have equal dimnesionality.')

        # dimension of each domain equal too number of states plus number of inputs
        if list(nxu)[0] != self.nx + self.nu:
            raise ValueError('the domains and the affine systems must have coherent dimensions.')

        # store inputs
        self.affine_systems = affine_systems
        self.domains = domains
        self.nm = len(affine_systems)
