from tensorflow.keras import Sequential
from tensorflow.keras.models import clone_model
from tensorflow.keras.models import Model as createModel
from tensorflow.keras.layers import Flatten,Conv1D,LSTM,MaxPooling1D,Embedding,Bidirectional, UpSampling2D,InputLayer, \
                                    Dropout, Dense, TimeDistributed,MaxPooling1D, MaxPooling1D,Reshape,Activation,   \
                                    Conv2D, MaxPooling2D, add,Conv1DTranspose, Conv2DTranspose, RepeatVector,BatchNormalization
from tensorflow.keras.layers import InputLayer as Input
import pyemd,json, sys, time, copy, os, copy


class DNN(object):
    def __init__(self,args=None):
        if args!=None:
            self.dataset_type = args.options.dataset_type

    def genomic_model(self,client_id=0):
        submodel = Sequential([
            Input((30,64,)),
            Bidirectional(LSTM(64,return_sequences=True,name="lstm1"),name="Bidirectional1"),
            Bidirectional(LSTM(32,name="lstm2"),name="Bidirectional2"),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(30)
        ])
        return submodel

    def imdb_model(self,client_id=0):
        max_features = 10000
        embedding_dim = 16

        submodel = Sequential([
            # Input((None,2494,)),
             # input_dim=len(encoder.get_vocabulary()),
            Embedding(input_dim=max_features + 1,output_dim=embedding_dim,mask_zero=True),
            Bidirectional(LSTM(64)),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        return submodel

    def image_model(self,client_id):
        if self.dataset_type=="cifar10":
            input_layer = Input((32,32,3))
        elif self.dataset_type=="fashion_mnist":
            input_layer = Input((28,28,1))

        submodel = Sequential([
            input_layer,
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            # Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(10)
        ])
        return submodel


    def image_model_large(self,client_id):
        if self.dataset_type=="cifar10":
            input_layer = Input((32,32,3))
            num_classes = 10
        elif self.dataset_type=="fashion_mnist":
            input_layer = Input((28,28,1))
            num_classes = 10

        submodel = Sequential([
            input_layer,
            Conv2D(32, 3, padding='same', activation='relu'),
            Conv2D(32, 3, activation='relu'),
            MaxPooling2D(),
            Dropout(0.25),

            Conv2D(64, 3, padding='same', activation='relu'),
            Conv2D(64, 3, activation='relu'),
            MaxPooling2D(),
            Dropout(0.25),

            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax'),
        ])
        return submodel

    def regression_model(self):
        input_layer = InputLayer((1))
        model_reg = Sequential([
        input_layer,
        Dense(2, activation='relu'),
        Dense(1)
        ])
        return model_reg

    def emnist(self,client_id):
        client_id = str(client_id)
        print("init EMNIST submodel  for client ",client_id)
        submodel = Sequential([
        Input((28, 28,1)                                                 ,name='input_'   + "_"+client_id),
        Conv2D(128, (5, 5),activation='relu',padding="same"),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Conv2D(64, (3, 3),activation='relu',padding="same"),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128                                          , activation='relu'),
        Dropout(0.5),
        Dense(62, activation='softmax'                                                  ,name='output_'  + "_"+client_id),
        ])
        return submodel

    def genomic_autoencoder(self):
        print("init mnist autoencoder ")
        read_size=30
        lstm_neurons=64
        cnn_neurons=128
        rate=0.5
        input_img = Input(shape=(read_size, 4,))
        x = Conv1D(cnn_neurons*1 ,kernel_size=5 ,activation='relu' ,padding='same' )(input_img)
        x = Conv1D(cnn_neurons*2 ,kernel_size=5 ,activation='relu' ,padding='same' )(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Dropout(rate)(x)
        x = LSTM(lstm_neurons, activation='tanh', return_sequences=True)(x)
        encoded = LSTM(30, activation='tanh')(x)

        x = RepeatVector(read_size)(encoded)
        x = LSTM(lstm_neurons, activation='tanh', return_sequences=True)(x)
        x = LSTM(lstm_neurons, activation='tanh', return_sequences=True)(x)

        x = Conv1DTranspose(cnn_neurons*2 ,kernel_size=5 ,activation='relu' ,padding='same' )(x)
        x = Conv1DTranspose(cnn_neurons*1 ,kernel_size=5 ,activation='relu' ,padding='same' )(x)
        decoded = Conv1DTranspose(4,3, activation='sigmoid', padding='same')(x)
        autoencoder = createModel(input_img,decoded)
        encoder = createModel(input_img, encoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        return autoencoder, encoder

    def cifar10_autoencoder(self):
        input_img = Input(shape=(32,32,3))

        #Encoder
        x = Conv2D(16,(3,3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2,2), padding='same')(x)

        x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2,2), padding='same')(x)

        x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2,2), padding='same', name='encoder')(x)

        #Decoder
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(16, (3, 3), activation='relu',padding='same')(x)
        x = UpSampling2D((2, 2))(x)

        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = createModel(input_img, decoded)
        encoder     = createModel(input_img, encoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder, encoder

    def fashion_mnist_autoencoder(self):
        encoded_size = 8
        inputs = Input(shape=(28, 28, 1))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, (2, 2), activation='relu', padding='same')(x)
        x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
        x = Conv2D(1, (2, 2), activation='relu', padding='same')(x)
        x = Flatten()(x)
        encoded = Dense(encoded_size, activation='relu')(x)

        encoded_inputs = Input(shape=(encoded_size,))

        x = Dense(4, activation='relu')(encoded_inputs)
        x = Reshape((2, 2, 1))(x)
        x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
        x = Conv2D(16, (2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((7, 7))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        encoder = createModel(inputs=inputs, outputs=encoded)
        decoder = createModel(inputs=encoded_inputs, outputs=decoded)

        x = encoder(inputs)
        x = decoder(x)
        autoencoder = createModel(inputs=inputs, outputs=x)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mse'])
        # autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder, encoder

    def cifar100_autoencoder(self):
        input_img = Input(shape=(32, 32, 3))
        x = Conv2D(64, (3, 3), padding='same')(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(16, (3, 3), padding='same')(encoded)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        decoded = Activation('sigmoid')(x)

        encoder     = createModel(inputs=input_img, outputs=encoded)
        autoencoder = createModel(input_img, decoded)
        # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        autoencoder.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
        return autoencoder, encoder

    def imdb_review_encoder(self):
        VOCAB_SIZE = 1000
        encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
        encoder.adapt(train_dataset.map(lambda text, label: text))
        model = tf.keras.Sequential([
            encoder,
            tf.keras.layers.Embedding(
                input_dim=len(encoder.get_vocabulary()),
                output_dim=64,
                mask_zero=True),
            ])
        return model

    def emnist_autoencoder(self):

        input_img = Input(shape=(28, 28, 1))
        x = Conv2D(32, (5, 5), activation='relu', padding='same',strides=1)(input_img)
        x = Conv2D(64, (5, 5), activation='relu', padding='same',strides=2)(x)
        encoded = Conv2D(2, (3, 3), activation='relu', padding='same', strides=2)(x)

        x = Conv2DTranspose(4, (3, 3), activation='relu', padding='same',strides=2)(encoded)
        x = Conv2DTranspose(64, (5,5), activation='relu', padding='same',strides=2)(x)
        x = Conv2DTranspose(32, (5,5), activation='relu', padding='same',strides=1)(x)
        decoded = Conv2DTranspose(1,(5,5), activation='sigmoid', padding='same')(x)

        autoencoder = createModel(input_img,decoded)
        encoder = createModel(input_img, encoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder, encoder

    def get_model(self,client_id,encoder=None):
        if self.dataset_type == "genomic":
            model = self.genomic_model(client_id=client_id)
        elif self.dataset_type == "cifar10" or self.dataset_type == "fashion_mnist":
            model = self.image_model_large(client_id)
        elif self.dataset_type == "imdb":
            model = self.imdb_model(client_id)
        elif self.dataset_type == "regression":
            model = self.regression_model()
        model._name=self.dataset_type + "_client_" + str(client_id)
        return model

    def get_autoencoder_model(self):

        if self.dataset_type == "cifar10":
            autoencoder, encoder = self.cifar10_autoencoder()
        elif self.dataset_type == "cifar100":
            autoencoder,encoder = self.cifar100_autoencoder()
        elif self.dataset_type == "mnist":
            autoencoder, encoder = self.mnist_autoencoder()
        elif self.dataset_type == "emnist":
            autoencoder, encoder = self.emnist_autoencoder()
        elif self.dataset_type == "fashion_mnist":
            autoencoder, encoder = self.fashion_mnist_autoencoder()
        elif self.dataset_type == "genomic":
            autoencoder, encoder = self.genomic_autoencoder()
        elif self.dataset_type == "nico":
            autoencoder, encoder = self.nico_autoencoder()
        elif self.dataset_type == "tf_flowers":
            autoencoder, encoder = self.tf_flowers_autoencoder()
        else:
            print("Wrong dataset type: EXIT")
            sys.exit(0)
        return autoencoder, encoder



    def flatten_weights(self,model_weights):
        weights_flatten=np.zeros(1)
        for _,w in enumerate(model_weights):
            try:
                weights_flatten = np.concatenate([weights_flatten,w.numpy().flatten()])
            except Exception:
                weights_flatten = np.concatenate([weights_flatten,w.flatten()])
        return weights_flatten
