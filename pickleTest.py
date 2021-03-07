import pickle
import sklearn.preprocessing.label

# store python objects in bytes (pickling or serialization)

# dict = {1: 'one', 2: 'two', 3: 'three'}

# pickling - write dict in bytes to pickle file
# pickle_out = open('dict.pickle', 'wb')
# pickle.dump(dict, pickle_out)
# pickle_out.close()

# unpickling - read dict in bytes from pickle file
# pickle_in = open('dict.pickle', 'rb')
# dict = pickle.load(pickle_in)

pickle_in = open('le.pickle', 'rb').read()
le = pickle.loads(pickle_in)

print(le)