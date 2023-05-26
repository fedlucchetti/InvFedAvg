from tensorflow.keras import Sequential
from tensorflow.keras.models import clone_model
from tensorflow.keras.models import Model as createModel
from tensorflow.keras.layers import Flatten,Conv1D,LSTM,MaxPooling1D,Embedding,Bidirectional, UpSampling2D,InputLayer, \
                                    Dropout, Dense,Input, TimeDistributed,MaxPooling1D, MaxPooling1D,Reshape,Activation,   \
                                    Conv2D, MaxPooling2D, add,Conv1DTranspose, Conv2DTranspose, RepeatVector,BatchNormalization
import pyemd,json, sys, time, copy, os, copy


class DraftFn(object):
    def __init__(self,args=None):
        if args!=None:
            self.dataset_type = args.options.dataset_type

    def get_model(n_neurons=2,n_layers=1):
        model_reg = keras.Sequential()
        model_reg.add(layers.Input(1))
        for id in range(n_layers):
            model_reg.add(layers.Dense(n_neurons, activation='relu'))
        model_reg.add(layers.Dense(1))
        model_reg.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))
        model_reg.summary()
        return model_reg

    def train_model(inputs,labels,n_neurons=2,n_layers=1,epochs=10,model=None):
        if model==None:
            model = get_model(n_neurons,n_layers)
        # model.build()
        history=model.fit(inputs,labels,validation_split=0.2, epochs=epochs)
        return model,history

    def train_models_diff_init(x,y,n_neurons=2,n_layers=1,epochs=60,n_models=2):
        models,id=list(),0
        import os
        while id!=n_models:
            model,h = train_model(x,y,n_neurons,n_layers,epochs)
            if h.history["val_loss"][-1]>0.07:
                print("----------------------------------------------")
                print("------------------ Continue ------------------")
                print("----------------------------------------------")
                continue
            else:
                models.append(model)
                os.system("clear")
                id+=1
                print("----------------------------------------------")
                print("------------------ OK id",id," ------------------")
                print("----------------------------------------------")
        return models

    def select_most_different_models(models):
        loss          = list()
        loss_fn       = tf.losses.MeanSquaredError()
        inputs        = np.random.uniform(-1,1,1000)
        inputs        = np.reshape(inputs,[inputs.size,1])
        for i in range(len(models)):
            for j in range(i+1,len(models)):
                labels = models[i](inputs)
                model  = models[j]
                model.compile(loss=loss_fn)
                l = model.evaluate(inputs,labels)
                loss.append([int(i),int(j),l])
        loss=np.array(loss)
        model_1 = models[int(loss[np.argmax(loss[:,2])][0])]
        model_2 = models[int(loss[np.argmax(loss[:,2])][1])]
        return model_1,model_2

    def save_models(models,client):
        utils.create_dir(join("workspace","saved_models_workspace"))
        for id,model in enumerate(models):
            path = join("workspace","saved_models_workspace","model_"+str(client)+"_"+str(id))
            print("saving to",path)
            model.save(path)

    def load_models(client,n=10):
        models= []
        for id in range(n):
            models.append(keras.models.load_model(join("workspace","saved_models_workspace","model_"+str(client)+"_"+str(int(id)))))
        return models

    def client_find_symmetric(submodel,dataset,client,alpha1=0.1,alpha2=0.001,c=4,epochs=1,batchsize=16):
        models=[]
        submodel_og = copy.deepcopy(submodel)
        models.append(submodel)
        del submodel
        # dataset = self.datasets_train[client]
        # dataset = dataset.batch(batchsize)
        x,y = dataset
        dataset = tensor2ds(dataset).batch(batchsize)
        loss_fn                 = tf.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        submodel_og.compile(loss=loss_fn,optimizer=optimizer)
        @tf.function
        def train_step(inputs,labels):
            with tf.GradientTape() as tape:
                predictions = submodel_og(inputs, training=True)
                pred_loss   = lr*self.loss_fn(labels, predictions)
            gradients   = tape.gradient(pred_loss, submodel_og.trainable_variables)
            optimizer.apply_gradients(zip(gradients, submodel_og.trainable_weights))
        for local_epoch in range(epochs):
            for i, (x, y) in enumerate(dataset):
                iter = i+1
                ti = 1/c*( (iter-1)%c + 1)
                if ti<=0.5: lr = (1-2*ti)*alpha1 + 2*ti*alpha2
                else:       lr = (2-2*ti)*alpha2 + (2*ti-1)*alpha1
                optimizer.learning_rate = lr
                train_step(x,y)
                if ti==1/2:
                    print(iter,ti,lr,"save")
                    models.append(submodel_og)
                    submodel_og.evaluate(self.Xval[client],self.Yval[client])
                    # del submodel_og
                    # submodel_og = copy.deepcopy(submodel)
                else:
                    print(iter,ti,lr,"boost lr")
        return models

    def train_all():
        model_central,_ = train_model(x,y,n_neurons=2,n_layers=1,epochs=60)
        models_1             = train_models_diff_init(x1,y1,n_neurons=2,n_layers=1,epochs=60,n_models=40)
        models_2             = train_models_diff_init(x2,y2,n_neurons=2,n_layers=1,epochs=60,n_models=40)
        save_models(models_1,1)
        save_models(models_2,2)
        model_1_A, model_1_B = select_most_different_models(models_1)
        model_2_A, model_2_B = select_most_different_models(models_2)
        model_1_A.save(join("workspace","saved_models_workspace","model_1_A"))
        model_1_B.save(join("workspace","saved_models_workspace","model_1_B"))
        model_2_A.save(join("workspace","saved_models_workspace","model_2_A"))
        model_2_B.save(join("workspace","saved_models_workspace","model_2_B"))
        model_central.save(join("workspace","saved_models_workspace","model_central"))
