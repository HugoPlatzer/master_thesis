from matplotlib import rc
import matplotlib.pyplot as plt

def apply_plot_settings():
    rc("text", usetex=True)
    rc("font", family="serif", size=11)
    rc("figure", figsize=(5.5, 4.5))
