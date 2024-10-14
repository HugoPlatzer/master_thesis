import plot_generator

pg = plot_generator.PlotGenerator("tests/test_plot_generator_1.json")
print(pg)
print(pg.json_data)

pg.generate_plot("plot.pdf")
