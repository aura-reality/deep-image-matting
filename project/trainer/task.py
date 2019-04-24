import argparse
import math
import os

import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model

from trainer.config import patience, batch_size, epochs, num_train_samples, num_valid_samples, checkpoint_models_path
from trainer.data_generator import train_gen, valid_gen
from trainer.migrate import migrate_model
from trainer.segnet import build_encoder_decoder, build_refinement
from trainer.utils import overall_loss, get_available_cpus, get_available_gpus
from trainer.model_checkpoint import MyModelCheckpoint, MyOtherModelCheckpoint

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pretrained", help="path to save pretrained model files")
    ap.add_argument("--job-dir", dest="job_dir", help="unused, but passed in by gcloud")

    args = vars(ap.parse_args())
    pretrained_path = args["pretrained"]

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = os.path.join(checkpoint_models_path, 'checkpoint.{epoch:02d}-{val_loss:.4f}.hdf5')
    model_checkpoint = MyModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)


    # Load our model, added support for Multi-GPUs
    num_gpu = len(get_available_gpus())
    if num_gpu >= 2:
        with tf.device("/cpu:0"):
            model = build_encoder_decoder()
            model = build_refinement(model)
            if pretrained_path is not None:
                model.load_weights(pretrained_path)
            else:
                migrate_model(model)

        final = multi_gpu_model(model, gpus=num_gpu)
        # rewrite the callback: saving through the original model and not the multi-gpu model.
        model_checkpoint = MyOtherModelCheckpoint(model, model_checkpoint)
    else:
        model = build_encoder_decoder()
        final = build_refinement(model)
        if pretrained_path is not None:
            final.load_weights(pretrained_path)
        else:
            migrate_model(final)

    decoder_target = tf.placeholder(dtype='float32', shape=(None, None, None, None))
    final.compile(optimizer='nadam', loss=overall_loss, target_tensors=[decoder_target])

    print(final.summary())

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    steps_per_epoch = math.ceil(num_train_samples / batch_size)
    validation_steps= math.ceil(num_valid_samples / batch_size)

    # Start Fine-tuning
    final.fit_generator(train_gen(),
                        steps_per_epoch=steps_per_epoch,
                        validation_data=valid_gen(),
                        validation_steps=validation_steps,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=2
                        )
