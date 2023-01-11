import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np 
import os.path as osp 
from utils.helpers import visualize, denormalize

def train_fit(model, epochs, optimizer, loss_fn, train_dataloader, val_dataloader, metrics=None, callbacks=None):
    model.compile(optimizer, loss_fn, metrics)

    history = model.fit(
        train_dataloader, 
        steps_per_epoch=len(train_dataloader), 
        epochs=epochs, 
        callbacks=callbacks, 
        validation_data=val_dataloader, 
        validation_steps=len(val_dataloader),
    )

    return history

def train_ctl(model, epochs, optimizer, loss_fn, train_dataloader, train_steps, val_dataloader, val_steps, val_dir, weights_dir, metrics=None, callbacks=None):

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x)
            # preds = tf.cast(preds, tf.float32)
            y = tf.cast(y, tf.float32)
            loss = loss_fn(y, preds)
            iou = metrics(y, preds)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return loss, iou

    @tf.function
    def validation_step(x, y):
        preds = model(x)
        y = tf.cast(y, tf.float32)
        test_loss = loss_fn(y, preds)
        test_iou = metrics(y, preds)

        return test_loss, test_iou

    TRAIN_LOSSES = []
    TRAIN_IOU_SCORES = []
    VAL_LOSSES = []
    VAL_IOU_SCORES = []
    best_iou_score = 0.0
    for epoch in range(epochs):
        # initializes training losses and iou_scores lists for the epoch
        losses = []
        iou_scores = []

        for step, (batch) in enumerate(train_dataloader):

            x, y = batch[0], batch[1]
            # run one training step for the current batch
            loss, iou = train_step(x, y)

            # Save current batch loss and iou-score
            losses.append(float(loss))
            iou_scores.append(float(iou))

            print("\r Epoch: {} >> step: {}/{} >> train-loss: {} >> IOU: {}".format(epoch, step, train_steps, \
                                np.round(sum(losses) / len(losses), 4), np.round(sum(iou_scores) / len(iou_scores), 4)), end="")

        # Save the train and validation losses and iou scores for each epoch.
        TRAIN_LOSSES.append(sum(losses) / len(losses))
        TRAIN_IOU_SCORES.append(sum(iou_scores) / len(iou_scores))
            
        if epoch % 1 == 0 and epoch != 0:
            print()
            val_losses = []
            val_iou_scores = []
            for val_step, val_batch in enumerate(val_dataloader):
                x_val, y_val = val_batch[0], val_batch[1]

                # val_loss, val_iou_score = distributed_val_step(x_val, y_val)
                val_loss, val_iou_score = validation_step(x_val, y_val)
                
                val_losses.append(val_loss)
                val_iou_scores.append(val_iou_score)

                print("** \rEpoch: {} >> step: {}/{} >> Val_Loss: {} >> Val_IOU-Score: {} ".format(epoch, val_step, val_steps, np.round(sum(val_losses) / len(val_losses), 4), \
                                    np.round(sum(val_iou_scores) / len(val_iou_scores), 4)), end="")
                
                    
            if sum(val_iou_scores) / len(val_iou_scores) > best_iou_score:
                best_iou_score = sum(val_iou_scores) / len(val_iou_scores)
                model.save_weights(osp.join(weights_dir, '{}_best_model.h5'))
                
                # preds = model(x_val)
                # visualize({"image" : denormalize(image.squeeze()), "gt_mask": y_val.squeeze(), \
                #     "pr_mask": preds.numpy().squeeze()}, fp=osp.join(vis_results, 'val_{}.png'.format(epoch)))

                for _val_step, _val_batch in enumerate(val_dataloader):
                    _x_val, _y_val = _val_batch[0], _val_batch[1]
                    _preds = model(_x_val)
                    visualize({"image" : denormalize(_x_val.squeeze()), "gt_mask": _y_val.squeeze(), \
                        "pr_mask": _preds.numpy().squeeze()}, fp=osp.join(val_dir, 'val_{}_{}.png'.format(epoch, _val_step)))
                print("------------------------------------------")
            VAL_LOSSES.append(sum(val_losses) / len(val_losses))
            VAL_IOU_SCORES.append(sum(val_iou_scores) / len(val_iou_scores))
            
        print()
        
    return TRAIN_IOU_SCORES, VAL_IOU_SCORES, TRAIN_LOSSES, VAL_LOSSES

