from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model, train_ds, val_ds, save_path="model.h5", epochs=10):
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(save_path, save_best_only=True)
    ]

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return model