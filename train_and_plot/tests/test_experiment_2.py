import experiment

e = experiment.Experiment("tests/test_experiment.json")
print(e)
e.run_training()
e.model.save_to_file("model.bin")
