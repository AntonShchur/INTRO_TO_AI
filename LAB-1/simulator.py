import numpy as np
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import terms
import tables
import control_system

class Simulator:
    def __init__(self, inputs, outputs, val_range, fn):
        # passed variables
        self.inputs = inputs
        self.outputs = outputs
        self.val_range = val_range
        self.fn = fn
        self.mem_fn = None
        # default values
        self.define_inner_variables()

    def simulate_variants(self, variants=[terms.trim_gen, terms.trap_gen, terms.gauss_gen]):
        self.__output_xy = [ self.fn(x, x) for x in self.__plot_range ]
        xy_avg = np.average(self.__output_xy)
        for mem_fn in variants:
            self.mem_fn = mem_fn
            first = True
            for diagonal in [False, True]:
                sys = self.__get_control_system(diagonal)
                self.__output_z = control_system.simulate2d(system=sys, input_range=self.__plot_range)
                title = mem_fn.__name__[:-4]
                if diagonal:
                    title += ' (diagonal)'
                    self.__calc_err(xy_avg, title)
                    self.__plot_results(first)
                    first = False
        self.mem_fn = None
        plt.show()
