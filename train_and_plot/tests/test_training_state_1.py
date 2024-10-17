import training_state

def test():
    ts = training_state.TrainingState(training_step=10, loss=4.5, accuracy=0.05)
    print(ts)
    print(ts.get_params())
