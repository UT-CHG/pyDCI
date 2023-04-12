# -*- coding: utf-8 -*-
"""
pyDCI Conftests

Fixtures for testing pyDCI classes and methods
"""

import shutil
from pathlib import Path
from scipy.stats.distributions import uniform, norm

import pandas as pd
import numpy as np
import pytest

from pydci.examples.monomial import monomial_1D
from pydci.utils import put_df

# from pydci.examples.


TEST_DIR = Path(__file__).parent / ".test_dir"


@pytest.fixture()
def test_dir():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(exist_ok=True)
    path = Path(TEST_DIR).absolute()
    yield path
    shutil.rmtree(path, ignore_errors=True)

@pytest.fixture
def monomial_1D_DCI():
    np.random.seed(123)

    def get_data(p, n_samples=int(1e3), N=1, mean=0.25, std_dev=0.1):
        lam, q_lam, data, std_dev = monomial_1D(p, n_samples=n_samples,
                                                N=N, mean=mean, std_dev=std_dev)
        pi_obs = norm(data, scale=std_dev)
        return lam, q_lam, pi_obs

    return get_data

@pytest.fixture
def monomial_1D_DCI_df():
    np.random.seed(123)

    def get_data(p, n_samples=int(1e3), N=1, mean=0.25, std_dev=0.1):
        lam, q_lam, data, std_dev = monomial_1D(p, n_samples=n_samples,
                                                N=N, mean=mean, std_dev=std_dev)
        pi_obs = norm(data, scale=std_dev)
        samples = pd.DataFrame(lam, columns=['lam_0'])
        samples = put_df(samples, "q_lam", q_lam)

        return samples, pi_obs

    return get_data


@pytest.fixture
def monomial_1D_MUD():
    np.random.seed(123)

    p, n_samples, data, std_dev = 5, int(1e3), [0.25], 0.1
    lam = uniform.rvs(size=(n_samples, 1), loc=-1, scale=2)
    q_lam = (lam**p).reshape(n_samples, -1)

    return lam, q_lam, data, std_dev

