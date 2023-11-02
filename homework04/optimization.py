import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict
from scipy.linalg import cholesky, cho_solve, cho_factor
from scipy.optimize.linesearch import scalar_search_wolfe2
from scipy.optimize import line_search
import time


class LineSearchTool(object):

    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        def phi_alpha(alpha_local):
            return oracle.func_directional(x_k, d_k, alpha_local)
        def phi_grad_alpha(alpha_local):
            return oracle.grad_directional(x_k, d_k, alpha_local)
        if self._method == 'Wolfe':
            try:
                alpha = scalar_search_wolfe2(phi=phi_alpha, derphi=phi_grad_alpha, c1=self.c1, c2=self.c2)[0]
            except LineSearchWarning:
                alpha = None
            if alpha is None:
                alpha = self.armijo_line_search(oracle, x_k, d_k, previous_alpha)
        elif self._method == 'Armijo':
            alpha = self.armijo_line_search(oracle, x_k, d_k, previous_alpha)
        elif self._method == 'Constant':
            alpha = self.c
        return alpha

    def armijo_line_search(self, oracle, x_k, d_k, previous_alpha=None):
        alpha = previous_alpha if previous_alpha is not None else self.alpha_0
        while True:
            f_k = oracle.func(x_k)
            grad_k = oracle.grad(x_k)
            x_new = x_k + alpha * d_k
            f_new = oracle.func(x_new)
            armijo_condition = f_new <= f_k + self.c1 * alpha * np.dot(grad_k, d_k)
            if armijo_condition:
                return alpha
            alpha *= 0.5
        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    def count_norm_grad(x):
        grad_x = oracle.grad(x)
        return np.sum(np.square(grad_x)) ** 0.5

    def make_history(x, time_work):
        history["func"] = history.get("func", []) + [oracle.func(x)]
        history["grad_norm"] = history.get("grad_norm", []) + [count_norm_grad(x)]
        history["time"] = history.get("time", []) + [time_work]
        if x.size <= 2:
            history["x"] = history.get("x", []) + [x]

    def make_display(x):
        print("x_k =", x)

    iter_count = 0
    if trace:
        make_history(x_k, 0)
    if display:
        make_display(x_k)

    norm_start = count_norm_grad(x_0)

    time_start = time.time()

    while count_norm_grad(x_k) ** 2 > norm_start ** 2 * tolerance:
        if iter_count == max_iter:
            return x_k, "iterations_exceeded", history

        d_k = -oracle.grad(x_k)
        alpha = line_search_tool.line_search(oracle=oracle, x_k=x_k, d_k=d_k)
        x_0 = x_k
        x_k = x_0 - alpha * oracle.grad(x_0)
        iter_count += 1

        if trace:
            make_history(x_k, time.time() - time_start)
        if display:
            make_display(x_k)

        if None in x_k or sum(np.abs(x_k) > 10 ** 10) >= 1:
            return x_k, "computational_error", history

    return x_k, 'success', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    def count_norm_grad(x):
        grad_x = oracle.grad(x)
        return np.sum(np.square(grad_x)) ** 0.5

    def make_history(x, time_work):
        history["func"] = history.get("func", []) + [oracle.func(x)]
        history["grad_norm"] = history.get("grad_norm", []) + [count_norm_grad(x)]
        history["time"] = history.get("time", []) + [time_work]
        if x.size <= 2:
            history["x"] = history.get("x", []) + [x]

    def make_display(x):
        print("x_k =", x)

    iter_count = 0
    if trace:
        make_history(x_k, 0)
    if display:
        make_display(x_k)

    norm_start = count_norm_grad(x_0)
    time_start = time.time()

    while count_norm_grad(x_k) ** 2 > norm_start ** 2 * tolerance:
        if iter_count == max_iter:
            return x_k, "iterations_exceeded", history

        try:
            d_k = cho_solve(cho_factor(oracle.hess(x_k)), -oracle.grad(x_k))
        except LinAlgError:
            return x_k, 'newton_direction_error', history

        alpha = line_search_tool.line_search(oracle=oracle, x_k=x_k, d_k=d_k)
        x_0 = x_k
        x_k = x_0 + alpha * d_k
        iter_count += 1

        if trace:
            make_history(x_k, time.time() - time_start)
        if display:
            make_display(x_k)

        if None in x_k or sum(np.abs(x_k) > 10 ** 10) >= 1:
            return x_k, "computational_error", history

    return x_k, 'success', history