import sys

def print_progress(i: int, total: int, width: int = 20, stream=sys.stdout) -> None:
    """Print a one-line progress bar (~every 1/width of total)."""
    if total <= 0:
        return
    step = max(1, total // width)                          # ~every 5% if width=20
    if (i % step == 0) or (i == total - 1):
        if i >= total - 1:
            done, pct = width, 100
        else:
            done = (i * width) // total                    # 0..width
            pct  = done * (100 // width)                   # 0,5,10,...,95 (width=20)
        stream.write('\r[%-*s] %3d%%' % (width, '=' * done, pct))
        stream.flush()
        if i >= total - 1:
            stream.write('\n')
