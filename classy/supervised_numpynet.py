try:
    from NumPyNet.network import Network
    from NumPyNet.layers.connected_layer import Connected_layer
    from NumPyNet.layers.convolutional_layer import Convolutional_layer
    from NumPyNet.layers.maxpool_layer import Maxpool_layer
    from NumPyNet.layers.softmax_layer import Softmax_layer
    from NumPyNet.layers.dropout_layer import Dropout_layer
    from NumPyNet.layers.cost_layer import Cost_layer
    from NumPyNet.layers.cost_layer import cost_type
    from NumPyNet.layers.batchnorm_layer import BatchNorm_layer
    from NumPyNet.optimizer import Adam, SGD, Momentum
    from NumPyNet.utils import to_categorical
    from NumPyNet.utils import from_categorical
    from NumPyNet.metrics import mean_accuracy_score
    import numpy as np

    def accuracy (y_true, y_pred):
        '''
        Temporary metrics to overcome "from_categorical" missing in standard metrics
        '''
        from NumPyNet.metrics import mean_accuracy_score
        truth = from_categorical(y_true)
        predicted = from_categorical(y_pred)
        return mean_accuracy_score(truth, predicted)

    class NumPyNetImageNN(object):

        def __init__(self,*args):
            self.compiled=False
            self.args=args
        
        def fit(self,*args,**kwargs):
            images=args[0]
            im=images
            n_train=len(im.data)
            num_classes=len(im.target_names)
            if len(im.data[0].shape)==2:  # grayscale
                X=np.array(im.data)
                X = np.asarray([np.dstack((x, x, x)) for x in X])
            else:
                X=np.array(im.data)
                #
                
            # normalization to [0, 1]
            X = X/X.max()
            y = to_categorical(im.targets).reshape(n_train, 1, 1, -1)            
            

            epochs=kwargs.get('epochs',10)  
            num_samples = X.shape[0]
            batch=kwargs.get('batch',num_samples)


            if not self.compiled:
                self.compiled=True
                self.model = Network(batch=batch, input_shape=X.shape[1:])

                for arg in self.args:
                    self.model.add(arg)

                self.model.compile(optimizer=Adam(), metrics=[accuracy])
                self.model.summary()
     

            self.model.batch=batch
            self.model.fit(X=X, y=y, max_iter=epochs)

            self.weights=[]
            self.bias=[]
            for layer in self.model._net:
                try:
                    w,b = layer.weights,layer.bias
                    self.weights.append(w)
                    self.bias.append(b)
                except AttributeError:  # Cost and input layers don't have weights
                    pass


        def percent_correct(self,im):
            n_train=len(im.data)

            predicted=self.predict(im)
            truth=im.targets
            performance=mean_accuracy_score(truth, predicted)

            return performance*100.0

        def output(self, im):
            n_train=len(im.data)
            num_classes=len(im.target_names)
            if len(im.data[0].shape)==2:  # grayscale
                X=np.array(im.data)
                X = np.asarray([np.dstack((x, x, x)) for x in X])
            else:
                X=np.array(im.data)
                
            # normalization to [0, 1]
            X = X/X.max()


            num_samples = X.shape[0]
            self.model.batch=num_samples
            _=self.model.predict(X=X,verbose=False)
  
            out=[]
            for layer in self.model:  
                out.append(layer.output[:].squeeze())

            return out

        def predict_names(self,im):
            result=self.predict(vectors)
            return [im.target_names[i] for i in result]


        def predict(self, im):
            n_train=len(im.data)
            num_classes=len(im.target_names)
            if len(im.data[0].shape)==2:  # grayscale
                X=np.array(im.data)
                X = np.asarray([np.dstack((x, x, x)) for x in X])
            else:
                X=np.array(im.data)
                
            # normalization to [0, 1]
            X = X/X.max()
            y = to_categorical(im.targets).reshape(n_train, 1, 1, -1)            


            num_samples = X.shape[0]
            self.model.batch=num_samples
            out=self.model.predict(X=X,verbose=False)
            predicted = from_categorical(out)
            
            return predicted.ravel()




    class NumPyNetBackProp(object):
        def __init__(self,model_dict=None,**kwargs):
            self.model_dict=model_dict
            self.model=None

            self.dummy_y=None


        def fit(self,*args,**kwargs):
            X,y=args[0],args[1]
            epochs=kwargs.get('epochs',1000)
            #
            
            # Reshape the data according to a 4D tensor
            num_samples, size = X.shape
            one_hot_y=to_categorical(y)
            num_classes=one_hot_y.shape[1]

            X = X.reshape(num_samples, 1, 1, size)
            self.dummy_y = one_hot_y = one_hot_y.reshape(num_samples,1,1,-1)
            batch=kwargs.get('batch',num_samples)


            if self.model is None:
                if self.model_dict is None:
                    self.model_dict = {
                                        'input':size,               # number of features
                                        'hidden':[(5,'logistic'),],
                                        'output':(num_classes,'logistic'),  # number of classes
                                        'cost':'mse',
                                    }
                self.model = Network(batch=batch, input_shape=(1,1,self.model_dict['input']))

                hidden=self.model_dict.get('hidden',[])
                for n,typ in hidden:
                    self.model.add(Connected_layer(outputs=n, activation=typ))

                n,typ=self.model_dict['output']
                self.model.add(Connected_layer(outputs=n, activation=typ))
                self.model.add(Cost_layer(cost_type=self.model_dict['cost']))
                self.model.compile(optimizer=Adam(), metrics=[accuracy])
                self.model.summary()

            self.model.batch=batch
            self.model.fit(X=X, y=one_hot_y, max_iter=epochs)
         
        
            self.weights=[]
            self.bias=[]
            for layer in self.model._net:
                try:
                    w,b = layer.weights,layer.bias
                    self.weights.append(w)
                    self.bias.append(b)
                except AttributeError:  # Cost and input layers don't have weights
                    pass


        
        def percent_correct(self,vectors,targets):

            from NumPyNet.metrics import mean_accuracy_score
            predicted=self.predict(vectors)
            truth=targets
            performance=mean_accuracy_score(truth, predicted)

            return performance*100.0
        

        def output(self, X):
            # Reshape the data according to a 4D tensor
            num_samples, size = X.shape
            X = X.reshape(num_samples, 1, 1, size)
            self.model.batch=num_samples
            _=self.model.predict(X=X,verbose=False)

            out=[]
            for layer in self.model:  # no input or cost 
                if isinstance(layer,Connected_layer):
                    out.append(layer.output[:].squeeze())


            return out

        def predict(self, X):
            from NumPyNet.utils import from_categorical

            # Reshape the data according to a 4D tensor
            num_samples, size = X.shape
            X = X.reshape(num_samples, 1, 1, size)
            self.model.batch=num_samples
            out=self.model.predict(X=X,verbose=False)
            predicted = from_categorical(out)
            
            return predicted.ravel()
        
        def predict_names(self,vectors,names):
            result=self.predict(vectors)
            return [names[i] for i in result]






except ImportError:
    class NumpyNetMLP(object):
        def __init__(self,**kwargs):
            raise NotImplementedError("NumpyNet not installed")    


