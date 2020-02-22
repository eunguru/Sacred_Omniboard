from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('hello_config')
# jupyter execute
# @ex.automain (x), @ex.main (o) 
# ex = Experiment('hello_config', interactive=True)

# MongoObserver
ex.observers.append(MongoObserver(url='localhost:27017', 
                                db_name='MY_DB'))

@ex.config
def my_config():
    recipient = "world"
    message = "Hello %s!" % recipient

@ex.automain
def my_main(message):
    print(message)
