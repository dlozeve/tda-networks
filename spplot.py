#!/usr/bin/env python3

import numpy as np
import igraph as ig

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (10, 6)

if __name__=="__main__":
    g = ig.read("data/sociopatterns/infectious/infectious.graphml")
    g.simplify()
    layout = g.layout_grid_fruchterman_reingold()
    # layout = g.lgl()
    ig.plot(g, layout=layout)
    
