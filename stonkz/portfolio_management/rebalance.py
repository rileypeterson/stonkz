import numpy as np


def no_sell_reallocate(x, t, limit=None):
    """
    Optimally (I hope so) realign the current values of investments x to some
    portfolio target allocation t, with the limitation that you cannot reduce the
    amount of any current investment (i.e. you can't sell).

    Parameters
    ----------
    x : array-like
        Current dollar amounts of each asset allocation.
    t : array-like
        The desired target allocation. Expressed as a decimal between [0, 1].
    limit : None | int | float, optional
        The maximum amount of new money you can afford to put in.

    Returns
    -------
    np.array
        The amount to buy of each asset, in order to reach the target allocation (or get as close as possible).

    """
    assert len(x) == len(t), "Number of current assets must equal number of targets"
    assert all(i <= 1 for i in t), "Targets should be less than or equal to 1"
    assert sum(t) == 1, "Asset allocation targets should sum to 1"
    x = np.array(x)
    t = np.array(t)
    s0 = sum(x)
    nan_t = t.copy()
    nan_t[t == 0] = np.nan
    div = x / nan_t
    print(div)
    s = np.nanmax(div) + np.sum(x[t == 0])
    i = np.argmax(div) + 1
    b = np.array([s0] + [-j for j in x])
    a = np.eye(len(x) + 1)
    a[0, 1:] = -np.ones(len(x))
    a[1:, 0] = -np.array(t)
    a = np.delete(a, i, 0)
    a = np.delete(a, i, 1)
    b = np.delete(b, i, 0)
    a[0, 1:] = 0
    b[0] = s
    a = np.vstack((a, [1] + (a.shape[1] - 1) * [-1]))
    b = np.append(b, s0)
    out = np.linalg.lstsq(a, b, rcond=None)
    out = out[0]
    print(out)
    if limit and limit + s0 < out[0]:
        pcts = out[1:] / sum(out[1:])
        out = limit * pcts
        out = np.insert(out, 0, sum(x) + sum(out))
    out = np.insert(out, i, 0)
    print(f"New Account Total: ${out[0]:.2f}")
    out = np.delete(out, 0)
    for i, amt in enumerate(out):
        print(f"x{i} Amount to Buy: ${amt:.2f}")
    new_allocs = list(((x + out) / (sum(x) + sum(out))))
    new_allocs = [float(f"{k:.2f}") for k in new_allocs]
    print(f"New Allocation: {new_allocs}")
    print(f"New Allocation Amounts: {list((x + out).round(2))}")

    return out.round(2)


if __name__ == "__main__":
    x = [5000, 1000, 2500, 100]
    t = [0.5, 0.15, 0.34, 0.01]
    print(no_sell_reallocate(x, t))
