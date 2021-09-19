import pandas as pd

from utils.model import Perceptron
from utils.all_utils import prepare_data


AND = {
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "y": [0, 0, 0, 1]
}


df = pd.DataFrame(AND)

X, y = prepare_data(df)

lr = 0.3
EPOCHS = 10
model = Perceptron(lr=lr, epochs=EPOCHS)
model.fit(X, y)
model.total_loss()
