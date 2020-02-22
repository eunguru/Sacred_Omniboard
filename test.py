from sacred import Experiment

# create experiment instance
ex = Experiment()

# decorate
@ex.automain
def my_main():
    print("Hello world")