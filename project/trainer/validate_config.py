import trainer.config as config

if not config.composite_backgrounds and config.loss_ratio != 1:
    raise Exception("If not compositing backgrounds, only alpha_loss should be used!")
