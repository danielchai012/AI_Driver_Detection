import pickle
import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
print(os.path.dirname(__file__))
with open(os.path.join(os.path.dirname(__file__), 'encodings.pickle'), 'rb') as f:
    new_dict = pickle.load(f)

print(new_dict)
