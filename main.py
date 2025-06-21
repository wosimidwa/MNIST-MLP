import tensorflow as tf
from data.load_data import get_data
from models.simple_mlp import build_model
from training.train_fit import train_with_fit
from training.train_manual import train_manually

if __name__ == "__main__":
    data = get_data()
    model = build_model()
    train_with_fit(model, data, epochs=5)
    test_model()
   
