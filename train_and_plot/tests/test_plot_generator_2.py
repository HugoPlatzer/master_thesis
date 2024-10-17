import plot_generator

def test():
    pg = plot_generator.PlotGenerator("tests/test_plot_generator_2.json")
    
    pg.generate_plot("plot.pdf")
