import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, SimpleRNN, Reshape, Input, concatenate, LSTM
from keras.optimizers import Adam
from keras import backend as K


class LinearModel(tf.keras.Model):

    def __init__(self):
        super(LinearModel, self).__init__()
        lr = 0.0001
        loss = tf.keras.losses.MeanSquaredError()

        rain_input = Input(shape=(1,), name='rain_input')

        paderno_input = Input(shape=(1,), name='paderno_input')

        niguarda_input = Input(shape=(1,), name='niguarda_input')

        niguarda_input_minus_1 = Input(shape=(1,), name='niguarda_input_minus_1')

        cantu_input = Input(shape=(1,), name='cantu_input')

        x = concatenate([rain_input, paderno_input, niguarda_input, niguarda_input_minus_1, cantu_input])

        # We stack a deep densely-connected network on top
        x = Dense(20, activation='relu')(x)
        x = Dense(40, activation='relu')(x)
        x = Dense(80, activation='relu')(x)
        x = Dense(40, activation='relu')(x)
        x = Dense(20, activation='relu')(x)

        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='linear', name='main_output')(x)

        optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

        self.model = Model(inputs=[rain_input, paderno_input, niguarda_input, niguarda_input_minus_1, cantu_input], outputs=main_output)

        mselog = tf.keras.losses.MeanSquaredLogarithmicError()

        def mean_diff_pred(y_true, y_pred):
            return K.mean(K.abs(y_pred - y_true))

        def max_diff_pred(y_true, y_pred):
            return K.max(K.abs(y_pred - y_true))

        def coeff_determination(y_true, y_pred):
            ss_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
            ss_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
            return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())

        self.model.compile(optimizer, loss=loss, metrics=[mselog, coeff_determination, mean_diff_pred, max_diff_pred],
                           loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)

        self.model.summary()

    def fit(self, x, y, epochs, verbose, validation_split, validation_data):
        history = tf.keras.callbacks.History()

        reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto',
                                                    min_delta=0.0001, cooldown=0, min_lr=0)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0,
                                                      mode='auto', baseline=None, restore_best_weights=True)

        history_file = self.model.fit(x=x, y=y, batch_size=None, epochs=epochs, verbose=verbose, callbacks=[early_stop, history], validation_split=validation_split, validation_data=validation_data, shuffle=True)

        return history_file

    def predict(self, x):
        predictions = self.model.predict(x=x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1,
                use_multiprocessing=False)
        return predictions
