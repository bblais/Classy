import jax
import jax.numpy as jnp               # JAX NumPy

from flax import linen as nn          # The Linen API
from flax.training import train_state
import optax                          # The Optax gradient processing and optimization library

import numpy as np                    # Ordinary NumPy
from tqdm import tqdm

class Flatten(object):
    def __init__(self):
        pass

    def __call__(self,x):
        return x.reshape((x.shape[0], -1))

class ReLU(object):
    def __init__(self):
        pass

    def __call__(self,x):
        return nn.relu(x)

class Sigmoid(object):
    def __init__(self):
        pass

    def __call__(self,x):
        return nn.sigmoid(x)

class Average_Pool(object):
    def __init__(self,window_shape=(2, 2), strides=(2, 2),**kwargs):
        self.dict=kwargs
        self.dict['window_shape']=window_shape
        self.dict['strides']=strides
        
    def __call__(self,x):
        return nn.avg_pool(x, **self.dict)


from flax.linen import Conv as Convolutional
from flax.linen import Dense



        
    
class Input(object):

    def __init__(self,shape):
        self.shape=shape


from flax.errors import ScopeParamShapeError

class NeuralNetwork(nn.Module):
    layers:tuple
    
    @nn.compact
    def __call__(self, x):
        try:
            for i,layer in enumerate(self.layers):
                x=layer(x)
        except ScopeParamShapeError:
            print("Shape Error on Layer %d: %s." % (i,str(layer)))
            raise
            
        return x

    def get_layer_output(self, x, layer_idx):
        """Safely apply a specific layer by calling it inside a method."""
        return self.layers[layer_idx](x)



def compute_metrics(logits, labels,num_classes):
  loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, num_classes=num_classes)))
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy
  }
  return metrics

from functools import partial
@partial(jax.jit, static_argnums=(2,3))
def train_step(state, batch,num_classes: int,layers:tuple):
  def loss_fn(params):
    logits = NeuralNetwork(layers).apply({'params': params}, batch['image'])
    loss = jnp.mean(optax.softmax_cross_entropy(
        logits=logits,
        labels=jax.nn.one_hot(batch['label'], num_classes=num_classes)))
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, batch['label'],num_classes)
  return state, metrics    

@partial(jax.jit, static_argnums=(2,3))
def eval_step(params, batch,num_classes,layers):
  logits = NeuralNetwork(layers).apply({'params': params}, batch['image'])
  return compute_metrics(logits, batch['label'],num_classes=num_classes)

def eval_model(model, vectors,targets,num_classes,layers):
    batch={'image':vectors,'label':targets}    
    metrics = eval_step(model, batch,num_classes,layers)
    metrics = jax.device_get(metrics)
    eval_summary = jax.tree.map(lambda x: x.item(), metrics)
    return eval_summary['loss'], eval_summary['accuracy']    


def train_epoch(state, vectors,targets,num_classes,batch_size, epoch, rng,layers):
    train_ds_size = len(vectors)
    steps_per_epoch = train_ds_size // batch_size
    
    perms = jax.random.permutation(rng, len(vectors))
    perms = perms[:steps_per_epoch * batch_size]  # Skip an incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    batch_metrics = []
    
    for perm in perms:
        batch={'image':vectors[perm,...],
               'label':targets[perm]
              }
        state, metrics = train_step(state, batch,num_classes,layers)
        batch_metrics.append(metrics)

    training_batch_metrics = jax.device_get(batch_metrics)
    training_epoch_metrics = {
          k: np.mean([metrics[k] for metrics in training_batch_metrics])
          for k in training_batch_metrics[0]}

    #print('Training - epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, training_epoch_metrics['loss'], training_epoch_metrics['accuracy'] * 100))

    return state, training_epoch_metrics




class BackProp(object):

    def __init__(self,layers,learning_rate=0.001,):

        self.rng = jax.random.PRNGKey(0)
        self.rng, self.init_rng = jax.random.split(self.rng)

        
        self.mlp = NeuralNetwork(tuple(layers[1:]))
        self.input=layers[0]
        self.full_shape=[1]+list(self.input.shape)

        if len(self.input.shape)==2:  # 2D images
             self.full_shape+=[1]
        
        self.params = self.mlp.init(self.init_rng, jnp.ones(self.full_shape))['params']

        self.tx = optax.adam(learning_rate=learning_rate)
        self.state = train_state.TrainState.create(apply_fn=self.mlp.apply, params=self.params,tx=self.tx)

        self.batch_size=64
        self.training_losses=[]
        self.training_accuracies=[]

        print(self)
        
    def __repr__(self):
        return self.mlp.tabulate(self.init_rng, jnp.ones(self.full_shape),
                   compute_flops=False, compute_vjp_flops=False)

    def output(self,x):
        y, state = self.mlp.apply({'params': self.state.params}, jnp.atleast_2d(jnp.array(x)), 
                    capture_intermediates=True, mutable=["intermediates"])
        intermediates = state['intermediates']
    
        arr=[]
        for key in intermediates:
            if isinstance(intermediates[key],dict):
                arr.append(intermediates[key]['__call__'])


        return arr



    def fit(self,vectors,targets,epochs=10,verbose=False):

        
        num_classes=self.mlp.layers[-1].features
        if len(vectors.shape)==3:  # images
            shape=list(vectors.shape)+[1]
            vectors=jnp.reshape(vectors,shape)
        
        for epoch in tqdm(range(1, epochs + 1)):
            # Use a separate PRNG key to permute image data during shuffling
            self.rng, self.input_rng = jax.random.split(self.rng)
            # Run an optimization step over a training batch
            self.state, self.train_metrics = train_epoch(self.state, 
                                                       vectors,targets,
                                                       num_classes,
                                                       self.batch_size, 
                                                       epoch, 
                                                       self.input_rng,
                                                       self.mlp.layers)
            # # Evaluate on the test set after each training epoch
            # test_loss, test_accuracy = eval_model(self.state.params, 
            #                                     data_test,
            #                                     num_classes,
            #                                     mlp.layers)

            training_loss=self.train_metrics['loss']
            training_accuracy=self.train_metrics['accuracy']
            if verbose:
                print('Testing - epoch: %d, loss: %.2f, accuracy: %.2f' % (epoch, 
                                                                           test_loss, 
                                                                           test_accuracy * 100))
            # Store metrics for graph visualization
            self.training_losses.append(training_loss)
            self.training_accuracies.append(training_accuracy)
            # testing_losses.append(test_loss)
            # testing_accuracies.append(test_accuracy)
        

    def percent_correct(self,vectors,targets):
        num_classes=self.mlp.layers[-1].features  
        if len(vectors.shape)==3:  # images
            shape=list(vectors.shape)+[1]
            vectors=jnp.reshape(vectors,shape)
        
        test_loss, test_accuracy = eval_model(self.state.params, 
                                            vectors,targets,
                                            num_classes,
                                            self.mlp.layers)
        return test_accuracy
