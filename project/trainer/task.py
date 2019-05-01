import argparse
import math
import os
from urllib.parse import urlparse

import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model

from trainer.config import patience, batch_size, epochs, num_train_samples, num_valid_samples, skip_crop
from trainer.data_generator import train_gen, valid_gen
from trainer.migrate import migrate_model
from trainer.segnet import build_encoder_decoder, build_refinement
from trainer.utils import overall_loss, get_available_cpus, get_available_gpus
from trainer.utils import alpha_prediction_loss
from trainer.model_checkpoint import MyModelCheckpoint, MyOtherModelCheckpoint

class Stage:

    def __init__(self, pretrained_path):
        self.pretrained_path = pretrained_path
        pass

    def get_loss(self):
        return overall_loss

    def get_target_tensors(self):
        return None

    def get_optimizer(self):
        return 'nadam'

    def get_model(self):
        pass

    def load_weights(self, model, allow_vgg=False):
        if not self.pretrained_path:
            if not allow_vgg:
                raise ValueError("Required a path to a pretrained model")
            else:
                #vgg
                migrate_model(model)
        else:
            print("Loading pretrained model weights")
            local_path = os.path.join('cache', urlparse(self.pretrained_path).path)
            cache(self.pretrained_path, local_path)
            model.load_weights(local_path)

class EncoderDecoderStage(Stage):

    def __init__(self, pretrained_path):
        super().__init__(pretrained_path)

    def get_model(self):
        model = build_encoder_decoder()
        self.load_weights(model, allow_vgg=True)
        return model

class RefinementStage(Stage):

    def __init__(self, pretrained_path):
        super().__init__(pretrained_path)

    def get_model(self):
        model = build_encoder_decoder()
        self.load_weights(model)

        # fix encoder-decoder part parameters and then update the refinement part.
        for layer in encoder_decoder.layers:
            layer.trainable = False

        self.model = build_refinement(model)
        return self.model

    # TODO why is this not implemented? can we just used overall_loss?
    # def get_loss():
    #     return custom_loss_wrapper(self.model.input)

class FinalStage(Stage):

    def __init__(self, pretrained_path):
        super().__init__(pretrained_path)

    # TODO: should we really be using alpha_predcition_loss here?
    # def get_loss():
    #    return alpha_prediction_loss

    def get_target_tensors(self):
        decoder_target = tf.placeholder(dtype='float32', shape=(None, None, None, None))
        return [decoder_target]

    def get_optimizer(self):
        return keras.optimizers.SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)

    def get_model(self):
        model = build_encoder_decoder()
        model = build_refinement(model)
        self.load_weights(model)

        # finetune the whole network together.
        for layer in model.layers:
            layer.trainable = True

        return model

class StagelessStage(Stage):

    def __init__(self, pretrained_path):
        super().__init__(pretrained_path)

    def get_target_tensors(self):
        decoder_target = tf.placeholder(dtype='float32', shape=(None, None, None, None))
        return [decoder_target]

    def get_model(self):
        model = build_encoder_decoder()
        model = build_refinement(model)
        self.load_weights(model, allow_vgg=True)
        return model

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pretrained", help="path to saved pretrained model")
    ap.add_argument("--job-dir", dest="job_dir", help="job-dir contains task module, checkpoint, logs")
    ap.add_argument("--stage", default="stageless", choices=["encoder_decoder",
                                                             "refinement",
                                                             "final",
                                                             "stageless"], help="the stage of training to run")

    args = vars(ap.parse_args())
    pretrained_path = args["pretrained"]
    job_dir = args["job_dir"]
    stage = args["stage"]

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir= job_dir + '/logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = os.path.join(job_dir + '/checkpoints', '%s.{epoch:02d}-{val_loss:.4f}.hdf5' % stage)
    model_checkpoint = MyModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)

    if stage == "encoder_decoder":
        stage = EncoderDecoderStage(pretrained_path)
    elif stage == "refinement":
        stage = RefinementStage(pretrained_path)
    elif stage == "final":
        stage = FinalStage(pretrained_path)
    elif stage == "stageless":
        stage = StagelessStage(pretrained_path)
    else:
        raise Exception("Unknown stage: %s" % stage)

    # Load our model, added support for Multi-GPUs
    num_gpu = len(get_available_gpus())
    if num_gpu >= 2:
        print("Building multi-gpu model")
        with tf.device("/cpu:0"):
            model = stage.get_model()

        # rewrite the callback: saving through the original model and not the multi-gpu model.
        model_checkpoint = MyOtherModelCheckpoint(model, model_checkpoint)
        model = multi_gpu_model(model, gpus=num_gpu)
    else:
        model = stage.get_model()

    optimizer = stage.get_optimizer()
    loss = stage.get_loss()
    target_tensors = stage.get_target_tensors()
    model.compile(optimizer=optimizer, loss=loss, target_tensors=target_tensors)

    print(model.summary())

    print("Running the '%s' stage" % stage)
    num_cpu = get_available_cpus()
    workers = int(round(num_cpu / 2))
    print('skip_crop={}\nnum_gpu={}\nnum_cpu={}\nworkers={}\ntrained_models_path={}.'.format(skip_crop, num_gpu, num_cpu, workers, model_names))

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    if batch_size > num_valid_samples: 
        print("Decreasing batch_size to %s to equal num_valid_samples" % batch_size)
        batch_size = num_valid_samples

    # Start Fine-tuning
    model.fit_generator(train_gen(),
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=valid_gen(),
                        validation_steps=num_valid_samples // batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=workers
                        )
