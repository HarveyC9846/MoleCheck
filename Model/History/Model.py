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
        ImagePart  = Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                            padding = "same", activation = "relu",
                            kernel_regularizer=regularizers.l2(0.01))(ImageInput)
        ImagePart  = MaxPooling2D(pool_size=(2,2))(ImagePart)
        ImagePart  = Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
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
        
    def CustomResNet(ImageSize, VariableSize):
        from keras.applications.resnet50 import ResNet50 as CreateResNet
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
        ResNet = CreateResNet(include_top=True, weights=None, input_tensor=None, input_shape=(64,64,3), pooling=None, classes=2)
        ResNet.layers.pop()

        ImageInput  = ResNet.input
        VariableInput = Input(shape=(VariableSize,))

        ModelNet = keras.layers.concatenate([ResNet.output, VariableInput])
        ModelNet = Dense(256, activation ='tanh', kernel_regularizer=regularizers.l2(0.01))(ModelNet)
        PredictOutput = Dense(2, activation ='softmax')(ModelNet)
        TheModel = Model(inputs=[ImageInput, VariableInput], outputs=PredictOutput)
        return(TheModel)

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
