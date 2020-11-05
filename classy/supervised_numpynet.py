try:
    from NumPyNet.network import Network
    from NumPyNet.layers.connected_layer import Connected_layer
    from NumPyNet.layers.cost_layer import Cost_layer
    from NumPyNet.optimizer import Adam,SGD
    from NumPyNet.utils import to_categorical
    from NumPyNet.utils import from_categorical
    from NumPyNet.metrics import mean_accuracy_score

    def accuracy (y_true, y_pred):
        '''
        Temporary metrics to overcome "from_categorical" missing in standard metrics
        '''
        from NumPyNet.metrics import mean_accuracy_score
        truth = from_categorical(y_true)
        predicted = from_categorical(y_pred)
        return mean_accuracy_score(truth, predicted)


    class NumPyNetBackProp(object):
        def __init__(self,**kwargs):
            self.model=None
            self.dummy_y=None
            self.hidden_layer_sizes=kwargs.get('hidden_layer_sizes',[4])

        def fit(self,*args,**kwargs):
            X,y=args[0],args[1]
            epochs=kwargs.get('epochs',1000)
            batch=batch_size=kwargs.get('batch_size',10)
            
            # Reshape the data according to a 4D tensor
            num_samples, size = X.shape
            one_hot_y=to_categorical(y)
            num_classes=one_hot_y.shape[1]

            X = X.reshape(num_samples, 1, 1, size)
            self.dummy_y = one_hot_y = one_hot_y.reshape(num_samples,1,1,-1)




            if self.model is None:

                self.model = Network(batch=batch, input_shape=X.shape[1:])

                for n in self.hidden_layer_sizes:
                    self.model.add(Connected_layer(outputs=n, activation='tanh'))

                self.model.add(Connected_layer(outputs=num_classes, activation='tanh'))
                self.model.add(Cost_layer(cost_type='mse'))
                self.model.compile(optimizer=Adam(), metrics=[accuracy])

                self.model.summary()


            model.fit(X=X, y=one_hot_y, max_iter=epochs)
         
        
            self.weights=[]
            self.bias=[]
            
            for layer in self.model.layers:
                w,b = layer.get_weights()     
                self.weights.append(w)
                self.bias.append(b)
        
        def percent_correct(self,vectors,targets):
            return self.score(vectors,targets)*100.0
        
        def percent_correct(self,vectors,targets):
            dummy_y = np_utils.to_categorical(targets)
            scores = self.model.evaluate(vectors, dummy_y,verbose=False)
            return scores[1]

        def output(self, X):
            return self.model.predict(X)

        def predict(self, X):
            output=self.model.predict(X)
            return np.argmax(output,axis=1)
        
        def predict_names(self,vectors,names):
            result=self.predict(vectors)
            return [names[i] for i in result]






except ImportError:
    class NumpyNetMLP(object):
        def __init__(self,**kwargs):
            raise NotImplementedError("NumpyNet not installed")    


