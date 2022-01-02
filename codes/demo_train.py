
from option import args
import data
import model
import utils
import loss
import trainer


if __name__ == '__main__':

    checkpoint = utils.checkpoint(args)
    if checkpoint.ok:
        dataloaders = data.create_dataloaders(args)
        sr_model = model.Model(args, checkpoint)
        sr_loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = trainer.Trainer(args, dataloaders, sr_model, sr_loss, checkpoint)

        while not t.terminate():
            t.train()
            t.test()

    checkpoint.done()
