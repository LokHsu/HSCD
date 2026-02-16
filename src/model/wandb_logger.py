from datetime import datetime

import wandb
from parser import parse_args
args = parse_args()


class WandbLogger():
    def __init__(self):
        self.best_perf = 0
        self.best_epoch = 0
        self.runs = f'HSCD-{datetime.now():%Y-%m-%d_%H-%M-%S}'
        wandb.init(
            project='HSCD',
            name=self.runs,
            config=args,
        )

    def log_metrics(self, epoch, loss, res):
        wandb.log({'epoch': epoch, 'loss': loss, 'hr': res['hr'], 'ndcg': res['ndcg']})
        if  self.best_perf < sum(res.values()):
            self.best_epoch = epoch
            self.best_perf = sum(res.values())
            wandb.run.summary['epoch(Best)'] = epoch
            wandb.run.summary['hr(Best)'] = res['hr']
            wandb.run.summary['ndcg(Best)'] = res['ndcg']

        if epoch == args.num_epochs:
            self.finish()

    def finish(self):
        wandb.finish()