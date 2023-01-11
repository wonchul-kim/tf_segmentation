import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np 
import os.path as osp 
from utils.helpers import visualize, denormalize

def train_ctl_multigpus(strategy, model, batch_size, epochs, optimizer, loss_fn, train_dist_dataset, val_dist_dataset, val_dir, weights_dir, metrics=None, callbacks=None):
    val_unscaled_loss = tf.keras.metrics.Sum(name='val_loss')
    val_unscaled_iou = tf.keras.metrics.Sum(name='val_iou')

    @tf.function
    def train_step(batch):
        image, label = batch 
        label = tf.cast(label, tf.float32)
        with tf.GradientTape() as tape:
            preds = model(image)
            loss = loss_fn(label, preds)
            loss = tf.reduce_sum(loss) * (1. / (batch_size*strategy.num_replicas_in_sync))
            # loss = tf.nn.compute_average_loss(loss,
            #                                     global_batch_size=self._vars.batch_size*self._strategy.num_replicas_in_sync)

            iou = metrics(label, preds)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # print("TRAINTRAIN -- unscaled_loss: ", loss, type(loss))
        # print(loss.numpy())
        return loss, iou

    @tf.function
    def distributed_train_epoch(ds):
        loss = 0.
        total_iou = 0.
        num_train_batches = 0.
        for batch in ds:
            per_replica_loss, per_replica_iou = strategy.run(train_step, args=(batch,))
            loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
            iou = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_iou, axis=None)
            loss += loss
            total_iou += iou

            num_train_batches += 1
        
        return loss, total_iou, num_train_batches

    @tf.function
    def val_step(batch):
        image, label = batch 
        label = tf.cast(label, tf.float32)
        preds = model(image)
        unscaled_loss = loss_fn(label, preds)
        unscaled_iou = metrics(label, preds)
        # print("VALVAL -- unscaled_loss: ", unscaled_loss, type(unscaled_loss))
        # print(unscaled_loss.numpy())
        val_unscaled_loss(unscaled_loss)
        val_unscaled_iou(unscaled_iou)

    @tf.function
    def distributed_val_epoch(ds):
        num_val_batches = 0.
        for batch in ds:
            # print("------------------------------------------------------------idx: ", idx)
            strategy.run(val_step, args=(batch,))
            # print("----------------------------------------------------------------")
            num_val_batches += 1

        return val_unscaled_loss.result(), val_unscaled_iou.result(), num_val_batches
        

    TRAIN_LOSSES = []
    TRAIN_IOU_SCORES = []
    VAL_LOSSES = []
    VAL_IOU_SCORES = []
    best_iou_score = 0.0
    best_val_loss = 999
    for epoch in range(epochs):
        train_loss, train_total_iou, num_train_batches = distributed_train_epoch(train_dist_dataset)
        val_loss, val_total_iou, num_val_batches = distributed_val_epoch(val_dist_dataset)
        train_avg_loss = train_loss / num_train_batches / strategy.num_replicas_in_sync
        train_avg_iou = train_total_iou / num_train_batches / strategy.num_replicas_in_sync

        val_avg_loss = val_loss / num_val_batches / strategy.num_replicas_in_sync
        val_avg_iou = val_total_iou / num_val_batches / strategy.num_replicas_in_sync

        print('>> Epoch: {}, Train Loss: {}, Train IoU: {}, Val Loss: {}, Val IoU: {}'.format(epoch, train_avg_loss, \
                                                                                    train_avg_iou, val_avg_loss, val_avg_iou))

        if best_val_loss > val_avg_loss:
            best_val_loss = val_avg_loss
            model.save_weights(osp.join(weights_dir, 'best_model.h5'))


        # Save the train and validation losses and iou scores for each epoch.
        TRAIN_LOSSES.append(train_avg_loss)
        TRAIN_IOU_SCORES.append(train_avg_iou)
        VAL_LOSSES.append(val_avg_loss)
        VAL_IOU_SCORES.append(val_avg_iou)
        
        val_unscaled_iou.reset_states()
        val_unscaled_loss.reset_states()


        # if sum(val_iou_scores) / len(val_iou_scores) > best_iou_score:
        #     best_iou_score = sum(val_iou_scores) / len(val_iou_scores)
        #     model.save_weights("files/model.h5")
