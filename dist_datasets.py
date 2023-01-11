import tensorflow as tf 
from src.datasets import Dataloader

def generator(dataloader, length_batch=2, use_multiprocessing=False):
    multi_enqueuer = tf.keras.utils.OrderedEnqueuer(dataloader, use_multiprocessing=use_multiprocessing)
    multi_enqueuer.start(workers=4, max_queue_size=10)
    for _ in range(len(dataloader)):
        batch_xs, batch_ys = next(multi_enqueuer.get())
        yield batch_xs, batch_ys

def get_dist_dataset(strategy, dataloader, length_batch=2, use_multiprocessing=False):
    _dataset = tf.data.Dataset.from_generator(lambda: generator(dataloader, length_batch, use_multiprocessing),
                                            output_types=(tf.float64, tf.float64),
                                            output_shapes=(tf.TensorShape([None, None, None, None]),
                                                            tf.TensorShape([None, None, None, None]))
                                            )
    dist_dataset = strategy.experimental_distribute_dataset(_dataset)

    return dist_dataset
