import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, concatenate, LSTM, Bidirectional
from keras.optimizers import Adam
from keras import backend as K


class LinearModel(tf.keras.Model):

    def __init__(self, n_steps, n_features):
        super(LinearModel, self).__init__()
        lr = 1e-5

        loss = tf.keras.losses.MeanSquaredError()

        input = Input(shape=(n_steps, n_features, ), name='input')

        x = Bidirectional(LSTM(30, activation='relu', return_sequences=True))(input)
        x = LSTM(30, activation='relu', return_sequences=True)(x)
        x = LSTM(30, activation='relu')(x)
        x = Dense(30, activation='relu')(x)
        x = Dense(30, activation='relu')(x)

        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='linear', name='main_output')(x)

        optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

        self.model = Model(inputs=input, outputs=main_output)

        mselog = tf.keras.losses.MeanSquaredLogarithmicError()

        def mean_diff_pred(y_true, y_pred):
            return K.mean(K.abs(y_pred - y_true))

        def max_diff_pred(y_true, y_pred):
            return K.max(K.abs(y_pred - y_true))

        def coeff_determination(y_true, y_pred):
            ss_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
            ss_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
            return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())

        self.model.compile(optimizer, loss=loss, metrics=[coeff_determination, mselog, mean_diff_pred, max_diff_pred],
                           loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)

        self.model.summary()

    def fit(self, x, y, validation_data, epochs, verbose):

        history = tf.keras.callbacks.History()

        reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0,
                                                        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0,
                                                      mode='auto', baseline=None, restore_best_weights=True)

        history_file = self.model.fit(x=x, y=y, steps_per_epoch=None, epochs=epochs, verbose=verbose,
                                    callbacks=[history, early_stop], validation_data=validation_data,
                                    validation_steps=None, shuffle=True)
        return history_file

    """
    def fit_generator(self, generator, validation_generator, epochs, verbose):

        history = tf.keras.callbacks.History()

        reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0,
                                                        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0,
                                                      mode='auto', baseline=None, restore_best_weights=True)

        history_file = self.model.fit_generator(generator=generator, steps_per_epoch=4928, epochs=10, verbose=verbose,
                                    callbacks=[history, early_stop], validation_data=validation_generator,
                                    validation_steps=1643, shuffle=False, max_queue_size=1, workers=1)

        return history_file
    """

    def predict(self, x):
        predictions = self.model.predict(x=x, batch_size=None, verbose=1, steps=None, callbacks=None, max_queue_size=10,
            workers=0, use_multiprocessing=False)
        return predictions

    def predict_generator(self, generator):
        predictions = self.model.predict_generator(generator=generator, steps=1643, callbacks=None, max_queue_size=10, workers=1,
                             use_multiprocessing=False, verbose=0)
        return predictions

