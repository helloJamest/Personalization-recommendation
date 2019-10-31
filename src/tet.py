import numpy as np

labels = [-1 ,1]
y = [np.random.choice(labels ,1)[0] for _ in range(10)]
x_field = [i // 10 for i in range(20)]
# x = np.random.randint(0 ,2 ,size=(10 ,input_x_size))
print(x_field)

