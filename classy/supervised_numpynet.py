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

    import json
    class NumpyAwareJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                if obj.ndim == 1:
                    return obj.tolist()
                else:
                    return [self.default(obj[i]) for i in range(obj.shape[0])]
            return json.JSONEncoder.default(self, obj)



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
            
            y = to_categorical(im.targets).reshape(n_train, 1, 1, -1)            
            

            epochs=kwargs.get('epochs',10)  
            num_samples = X.shape[0]
            batch=kwargs.get('batch',num_samples)
            learning_rate=kwargs.get('learning_rate',0.1)


            if not self.compiled:
                self.compiled=True
                self.model = Network(batch=batch, input_shape=X.shape[1:])

                for arg in self.args:
                    self.model.add(arg)

                self.model.compile(optimizer=Adam(), metrics=[accuracy])
                self.model.summary()
     

            self.model.batch=batch
            self.model.fit(X=X, y=y, max_iter=epochs,gamma=learning_rate)

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

        def init_model(self,batch):
            
            assert not self.model_dict is None
            if self.model is None:
                self.model = Network(batch=batch, input_shape=(1,1,self.model_dict['input']))

                hidden=self.model_dict.get('hidden',[])
                for n,typ in hidden:
                    self.model.add(Connected_layer(outputs=n, activation=typ))

                n,typ=self.model_dict['output']
                self.model.add(Connected_layer(outputs=n, activation=typ))
                self.model.add(Cost_layer(cost_type=self.model_dict['cost']))
                self.model.compile(optimizer=Adam(), metrics=[accuracy])
                self.model.summary()


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

            self.init_model(batch)

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

        def predict(self, X,**kwargs):
            from NumPyNet.utils import from_categorical

            # Reshape the data according to a 4D tensor
            num_samples, size = X.shape
            X = X.reshape(num_samples, 1, 1, size)

            batch=kwargs.get('batch',num_samples)

            self.model.batch=batch
            out=self.model.predict(X=X,verbose=False)
            predicted = from_categorical(out)
            
            return predicted.ravel()
        
        def predict_names(self,vectors,names):
            result=self.predict(vectors)
            return [names[i] for i in result]


        def save(self,filename):

            assert self.model._fitted
            
            D={}
            D['model_dict']=self.model_dict
            D['weights']=[L.weights if 'weights' in L.__dict__ else [] for L in self.model._net ]
            D['bias']=[L.bias if 'bias' in L.__dict__ else [] for L in self.model._net ]

            with open(filename, 'w') as f:
                json.dump(D,f, sort_keys=True, indent=4,cls=NumpyAwareJSONEncoder)        

        def load(self,filename):
            with open(filename, 'r') as f:
                D=json.load(f)
        
            self.model_dict=D['model_dict']
            self.init_model(1)

            for L,W,B in zip(self.model._net,D['weights'],D['bias']):
                if 'weights' in L.__dict__:
                    L.weights=np.array(W)
                    L.bias==np.array(B)

            self.model._fitted=True            


except ImportError:
    class NumpyNetMLP(object):
        def __init__(self,**kwargs):
            raise NotImplementedError("NumpyNet not installed")    


