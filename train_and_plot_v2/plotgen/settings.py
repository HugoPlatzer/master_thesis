from matplotlib import rc
import matplotlib.pyplot as plt

def apply_plot_settings():
    rc("text", usetex=True)
    rc("font", family="serif", size=10)
    rc("figure", figsize=(4.5, 3.5))
