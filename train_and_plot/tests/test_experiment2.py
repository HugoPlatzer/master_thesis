import experiment

def run_test(json_file):
    e = experiment.Experiment(json_file)
    print(e)
    e.run_training()
    e.model.save_to_file("model.bin")