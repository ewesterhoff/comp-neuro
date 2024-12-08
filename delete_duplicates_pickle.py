import pickle
import numpy as np

task = 'PDMa'
pickle_file = f'data/{task}_training_results.pkl'

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

print(len(data))

unique_data = []
deleted = 0
for i in range(len(data)):
    data1 = data[i]
    is_unique = True
    for j in range(i + 1, len(data)):
        data2 = data[j]

        try:
            if data1[0] == data2[0] and data1[1]['performance'] == data2[1]['performance'] and \
                    np.array_equal(data1[1]['W'], data2[1]['W']) and \
                    np.array_equal(data1[1]['evals'], data2[1]['evals']) and \
                    np.array_equal(data1[1]['evecs'], data2[1]['evecs']):
                is_unique = False
                deleted += 1
                break
        except Exception as e:
            # If an exception occurs, print data1, data2, and the exception message
            print("An error occurred:")
            print("data1:", data1)
            print("data2:", data2)
            print("Error:", e)
    if is_unique:
        unique_data.append(data1)


print(deleted)
# with open(pickle_file, 'wb') as f:
#     pickle.dump(unique_data, f)