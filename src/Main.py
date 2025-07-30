# Main.py
import argparse, json
from data.config import HP
from scripts.bayrak08 import bayrak08

def parse_cli() -> HP:
    p = argparse.ArgumentParser()
    p.add_argument('--lr',    type=float, default=HP.lr)
    p.add_argument('--batch', type=int,   default=HP.batch)
    p.add_argument('--epochs', type=int,  default=HP.epochs)
    ns = p.parse_args()
    return HP(**vars(ns))

if __name__ == "__main__":
    # Parse command line arguments
    cfg = parse_cli()
    print(cfg)
    
    # Execute simulation sweep from Bayrak & Bailey (2008)
    bayrak08(cfg)

