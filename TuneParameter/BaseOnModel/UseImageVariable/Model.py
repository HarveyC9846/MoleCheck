class Model:
    def I2CLMV1FLO(ImageSize, VariableSize):

        import keras
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Flatten
        from keras.layers import Conv2D, MaxPooling2D
        from keras.utils import to_categorical
        from keras.layers import Input
        from keras.models import Model
        from keras import regularizers
        from keras import backend
        import tensorflow
        import numpy

        ImageInput = Input(shape=ImageSize)
        ImagePart  = Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                            padding = "same", activation = "relu",
                            kernel_regularizer=regularizers.l2(0.01))(ImageInput)
        ImagePart  = MaxPooling2D(pool_size=(2,2))(ImagePart)
        ImagePart  = Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                            padding = "same", activation = "relu",
                            kernel_regularizer=regularizers.l2(0.01))(ImagePart)
        ImagePart  = MaxPooling2D(pool_size=(2,2))(ImagePart)
        ImagePart  = Flatten()(ImagePart)

        VariableInput = Input(shape=(VariableSize,))
        Pipe = keras.layers.concatenate([ImagePart, VariableInput])
        Pipe = Dense(1000, activation ='tanh',
                     kernel_regularizer=regularizers.l2(0.01))(Pipe)
        PredictOutput = Dense(2, activation ='softmax')(Pipe)
        model = Model(inputs=[ImageInput, VariableInput], outputs=PredictOutput)

        return(model)

    # def I2CL1FLO():
    #
    #     import keras
    #     from keras.models import Sequential
    #     from keras.layers import Dense, Dropout, Flatten
    #     from keras.layers import Conv2D, MaxPooling2D
    #     from keras.utils import to_categorical
    #     from keras.layers import Input
    #     from keras.models import Model
    #     from keras import regularizers
    #     from keras import backend
    #     import tensorflow
    #     import numpy
    #
    #     ImageInput = Input(shape=(32, 32, 3))
    #     ImagePart  = Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
    #                         padding = "same", activation = "relu",
    #                         kernel_regularizer=regularizers.l2(0.01))(ImageInput)
    #     ImagePart  = MaxPooling2D(pool_size=(2,2))(ImagePart)
    #     ImagePart  = Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
    #                         padding = "same", activation = "relu",
    #                         kernel_regularizer=regularizers.l2(0.01))(ImagePart)
    #     ImagePart  = MaxPooling2D(pool_size=(2,2))(ImagePart)
    #     ImagePart  = Flatten()(ImagePart)
    #
    #     Pipe = Dense(2000, activation ='tanh',
    #                  kernel_regularizer=regularizers.l2(0.01))(ImagePart)
    #     PredictOutput = Dense(2, activation ='softmax')(Pipe)
    #     model = Model(inputs=ImageInput, outputs=PredictOutput)
    #
    #     return(model)
