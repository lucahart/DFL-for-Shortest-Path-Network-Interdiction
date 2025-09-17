
import argparse
from dflintdpy.data.config import HP
from dflintdpy.scripts.asym_spni_single_sim import single_sim

def parse_cli() -> HP:
    p = argparse.ArgumentParser()
    p.add_argument('--num_seeds',    type=int, default=HP.num_seeds)
    p.add_argument('--pfl_epochs', type=int,   default=HP.po_epochs)
    p.add_argument('--dfl_epochs', type=int,   default=HP.spo_epochs)
    ns = p.parse_args()
    return HP(**vars(ns))


def main() -> None:
    # Parse command line arguments
    cfg = parse_cli()
    print(cfg)
    # Run the simulation
    single_sim(cfg)


if __name__ == "__main__":
    main()

