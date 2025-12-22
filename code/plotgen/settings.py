import matplotlib.pyplot as plt

# font size (in pt)
FONT_SIZE = 11
# size of the grid / curves area in a plot (in inches)
PLOT_GRID_SIZE = (5, 4)

def apply_font_settings():
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = FONT_SIZE
    plt.rcParams["font.family"] = "serif"
