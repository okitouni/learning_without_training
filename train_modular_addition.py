import itertools
import torch
from sklearn.model_selection import train_test_split
from lwot.models import GEMLinear, GEMEmbedding
from tqdm import trange
from utils import save_model
import wandb

class Model(torch.nn.Module):
  def __init__(self, vocab_size, d_model, masking=False):
    super().__init__()
    masking = bool(masking)
    train_weights = not masking
    train_scores = masking
    self.emb = GEMEmbedding(vocab_size, d_model, train_weights=train_weights, train_scores=train_scores)
    self.decoder = torch.nn.Sequential(
      GEMLinear(2*d_model, d_model, train_weights=train_weights, train_scores=train_scores),
      torch.nn.ReLU(),
      GEMLinear(d_model, d_model, train_weights=train_weights, train_scores=train_scores),
      torch.nn.ReLU(),
      GEMLinear(d_model, vocab_size, train_weights=train_weights, train_scores=train_scores),
    )


  def forward(self, x):
    x = self.emb(x)
    x = x.flatten(1)
    return self.decoder(x)

def train_modular_addition(args, device, basedir):
  """
  args: argparse.Namespace with the configuration in it
  device: torch.device
  basedir: str, path to the directory where the checkpoints will be saved
  """
  # modular addition dataset
  P = args.MODULO
  X = torch.tensor(list(itertools.combinations(range(P), 2)), dtype=torch.long, device=device)
  Y = X.sum(dim=1) % P

  X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=args.SEED)

  # define model
  model = Model(vocab_size=P, d_model=args.D_MODEL, masking=args.MASKING).to(device)

  # train model
  optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.WEIGHT_DECAY)
  bar = trange(args.EPOCHS)

  for epoch in bar:
    if args.WANDB and (epoch % (args.EPOCHS//20) == 0 or epoch == args.EPOCHS-1):
      save_model(basedir, model, epoch)
    yhat = model(X_train)
    loss = torch.nn.functional.cross_entropy(yhat, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
      acc = (yhat.argmax(dim=1) == Y_train).float().mean() * 100
      yhat = model(X_val)
      loss_val = torch.nn.functional.cross_entropy(yhat, Y_val)
      acc_val = (yhat.argmax(dim=1) == Y_val).float().mean() * 100
    bar.set_description(f"Train l: {loss:.3f}, a: {acc:.1f} | Val l: {loss_val:.3f}, a: {acc_val:.1f}")
