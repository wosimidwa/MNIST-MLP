#train and test model
def build_model():
  model = Sequential([
    Dense(256,activation='sigmoid'),
    Dense(128, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
])
  return model 
  
# %%

def train_with_fit(model, data, epochs=5):
    (x_train, y_train), (x_test, y_test) = data
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

# %% sonuç bakma zamanı

def test_model():
  plt.plot(mod.history['accuracy'], label='Training Accuracy')
  plt.plot(mod.history['val_accuracy'], label='Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.grid(True)
  plt.title('Model Performance')
  plt.show()

  test_loss, test_acc = model.evaluate(x_test, y_test)
  print(f"Test doğruluğu: {test_acc:.4f}")
