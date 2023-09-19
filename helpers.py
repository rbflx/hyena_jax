from typing import Callable
import jax
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax

from jax import lax, random, numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

import numpy as np

import os
import warnings
import functools
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def get_batch_gen(data_size: int, get_batch_items: Callable, batch_size: int, key: random.PRNGKey,
                  shuffle: bool = False, steps_per_epoch: int = None):
    """Create a new batch generator which yields steps_per_epoch batches. If steps_per_epoch is None,
    it is calculated as data_size // batch_size.

    Args:
        data_size: Total number of items to sample from.
        get_batch_items: Function that takes an array of indices as input and returns the associated items.
        batch_size: Batch size.
        key: PRNGKey used to sample random indices.
    
    Returns:
        batch_gen: Batch generator. Yields steps_per_epoch items.
        key: New PRNGKey.
    """
    steps_per_epoch = data_size // batch_size if steps_per_epoch is None else steps_per_epoch
    
    if shuffle:
        key, perm_key = random.split(key)
        perms = random.permutation(perm_key, data_size)
    else:
        perms = jnp.arange(data_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    def batch_gen():
        idx = 0
        while idx < steps_per_epoch:
            perm = perms[idx]
            batch = get_batch_items(perm)
            yield batch
            idx += 1

    return batch_gen, key


# Adapted from https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.ipynb
class TrainerModule:
    def __init__(self, model, model_name, exmp_batch, max_iters, seed=0, checkpoint_path='./saved_models'):
        """
        Inputs:
            model_name - Name of the model. Used for saving and checkpointing
            exmp_batch - Example input to the model for initialization
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            seed - Seed to use for model init
        """
        super().__init__()
        self.model_name = model_name
        self.max_iters = max_iters
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = model
        # Prepare logging
        self.checkpoint_path = checkpoint_path
        self.log_dir = os.path.join(self.checkpoint_path, self.model_name)
        self.should_save = True
        if os.path.exists(self.log_dir):
            warnings.warn('WARNING: Checkpoint directory already exists for this model. Model will not be saved.')
            self.should_save = False
        self.logger = SummaryWriter(log_dir=self.log_dir)
        self.init_state_and_funcs(exmp_batch)
    
    def init_state_and_funcs(self, exmp_batch):
        # Initialize model
        self.state = self.init_model(exmp_batch)
        # Create jitted training and eval functions
        self.create_functions()

    def get_loss_function(self):
        # Return a function that calculates the loss for a batch
        # To be implemented in a task specific sub-class
        raise NotImplementedError

    def init_model(self, exmp_batch):
        # Must return TrainState
        raise NotImplementedError

    def jit_train_step(self, train_step):
        return jax.jit(train_step)
    
    def jit_eval_step(self, eval_step):
        return jax.jit(eval_step)

    def create_functions(self):
        # Create jitted train and eval functions
        calculate_loss = self.get_loss_function()

        # Training function
        def train_step(state, rng, batch):
            loss_fn = lambda params: calculate_loss(params, rng, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, acc, rng = ret[0], *ret[1]
            state = state.apply_gradients(grads=grads)
            return state, rng, loss, acc

        self.train_step = self.jit_train_step(train_step)

        # Evaluation function
        def eval_step(state, rng, batch):
            loss, (acc, rng) = calculate_loss(state.params, rng, batch, train=False)
            return loss, acc, rng
        
        self.eval_step = self.jit_eval_step(eval_step)

    def train_model(self, get_train_gen, get_val_gen, num_epochs=500):
        # Train model for defined number of epochs
        best_acc = 0.0
        for epoch_idx in tqdm(range(1, num_epochs+1)):

            # Refresh generator for each epoch
            train_gen, self.rng = get_train_gen(key=self.rng)

            self.train_epoch(train_gen, epoch=epoch_idx)
            epoch_mod = 1
            if epoch_idx % epoch_mod == 0:
                val_gen, self.rng = get_val_gen(key=self.rng)
                eval_acc, avg_loss = self.eval_model(val_gen)
                self.logger.add_scalar('val/accuracy', eval_acc, global_step=epoch_idx)
                self.logger.add_scalar('val/loss', avg_loss, global_step=epoch_idx)
                print(f'Epoch {epoch_idx}, val loss: {avg_loss}, accuracy: {eval_acc}')
                if eval_acc >= best_acc:
                    best_acc = eval_acc
                    self.save_model(step=epoch_idx)
                self.logger.flush()

    def train_epoch(self, train_gen, epoch):
        # Train model for one epoch, and log avg loss and accuracy
        accs, losses = [], []
        for batch in tqdm(train_gen(), desc='Training', leave=False):
            self.state, self.rng, loss, accuracy = self.train_step(self.state, self.rng, batch)
            
            losses.append(loss)
            accs.append(accuracy)

        avg_loss = np.stack(jax.device_get(losses)).mean()
        avg_acc = np.stack(jax.device_get(accs)).mean()

        self.logger.add_scalar('train/loss', avg_loss, global_step=epoch)
        self.logger.add_scalar('train/accuracy', avg_acc, global_step=epoch)
        print(f'Epoch {epoch}, train loss: {avg_loss}, accuracy: {avg_acc}')

    def eval_model(self, data_gen):
        # Test model on all data points of a data loader and return avg accuracy
        losses = []
        accs = []

        for batch in data_gen():
            loss, acc, self.rng = self.eval_step(self.state, self.rng, batch)

            losses.append(loss)
            accs.append(acc)
        
        avg_loss = np.stack(jax.device_get(losses)).mean()
        avg_acc = np.stack(jax.device_get(accs)).mean()

        return avg_acc, avg_loss

    def save_model(self, step=0):
        # Save current model at certain training iteration
        if self.should_save:
            checkpoints.save_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, step=step)
        return

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for the pretrained model
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=self.state.params)
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.checkpoint_path, f'{self.model_name}.ckpt'), target=self.state.params)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx)


class ShardedTrainer(TrainerModule):
    def __init__(self, model, model_name, exmp_batch, max_iters, data_sharding, mesh_manager, seed=0, checkpoint_path='./saved_models'):
        self.data_sharding = data_sharding
        self.mesh_manager = mesh_manager
        super().__init__(model, model_name, exmp_batch, max_iters, seed, checkpoint_path)

    def init_state_and_funcs(self, exmp_batch):
        self.state, self.state_sharding = self.init_model(exmp_batch)
        self.create_functions()

    def jit_train_step(self, train_step):
        # state, rng, batch
        jitted_train_step = jax.jit(train_step,
                                    in_shardings=(self.state_sharding, None, self.data_sharding),   # state, rng, batch
                                    out_shardings=(self.state_sharding, None, None, None))  # state, rng, loss, acc
        return jitted_train_step
    
    def jit_eval_step(self, eval_step):
        jitted_eval_step = jax.jit(eval_step,
                                    in_shardings=(self.state_sharding, None, self.data_sharding),   # state, rng, batch
                                    out_shardings=(None, None, None))   # loss, acc, rng
        return jitted_eval_step


class MeshManager():
    """Helper class to provide sharding utilities for a given mesh.
    """
    def __init__(self, mesh):
        self.mesh = mesh

    def mesh_sharding(self, pspec: PartitionSpec) -> NamedSharding:
        """Create a NamedSharding from a given PartitionSpec applied to this instance's mesh.
        """
        return NamedSharding(self.mesh, pspec)

    def shard_data(self, data, named_sharding):
        """Shard a given array according to the sharding specification in named_sharding.
        """
        sharded_data = jax.device_put(data, named_sharding)
        return sharded_data

    def get_var_sharding(init_fn, *args, **kwargs):
        """Gets the output shape of init_fn when executing it with args and kwargs.
        args must be Pytrees that can be passed to jax.eval_shape. kwargs are all other
        arguments that can't be expressed as Pytrees (e.g. model, optimizer, etc.).

        Returns:
            abstract_vars: A nested PyTree containing jax.ShapeDtypeStruct objects as leaves.
            logical_spec: The PartitionSpec corresponding to abstract_vars.
        """

        abstract_vars = jax.eval_shape(
            functools.partial(init_fn, **kwargs), *args
        )
        logical_spec = nn.get_partition_spec(abstract_vars)

        return abstract_vars, logical_spec

    def logical_to_mesh(self, logical_spec, rules):
        """Convert a logical to a physical sharding according to provided rules.
        """
        dev_sharding = nn.logical_to_mesh_sharding(logical_spec, self.mesh, rules)
        return dev_sharding