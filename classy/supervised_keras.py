try:
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import np_utils
    import numpy as np
    
    class KerasMLP(object):
        
        def percent_correct(self,vectors,targets):
            return self.score(vectors,targets)*100.0
        def __init__(self,**kwargs):
            self.model=None
            self.dummy_y=None
            self.hidden_layer_sizes=kwargs.get('hidden_layer_sizes',[4])
        def fit(self,*args,**kwargs):
            X,Y=args[0],args[1]
            epochs=kwargs.get('epochs',300)
            batch_size=kwargs.get('batch_size',10)
            if self.model is None:
                self.model=Sequential()
                self.model.add(Dense(self.hidden_layer_sizes[0], input_dim=X.shape[1], activation='relu'))

                for n in self.hidden_layer_sizes[1:]:
                    self.model.add(Dense(n, activation='relu'))     
                    
                self.dummy_y = np_utils.to_categorical(Y)                
                self.model.add(Dense(self.dummy_y.shape[1], activation='sigmoid'))
                self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                
            self.model.fit(X, self.dummy_y,epochs=epochs,batch_size=batch_size,verbose=False)
         
        
            self.weights=[]
            self.bias=[]
            
            for layer in self.model.layers:
                w,b = layer.get_weights()     
                self.weights.append(w)
                self.bias.append(b)
        
        
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
    class KerasMLP(object):
        def __init__(self,**kwargs):
            raise NotImplementedError("Keras not installed")