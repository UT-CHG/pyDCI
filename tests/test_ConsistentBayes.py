# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats.distributions import uniform, norm
from scipy.stats import gaussian_kde as gkde  # type: ignore

from pydci.utils import put_df
from pydci import ConsistentBayes as CB

def test_DCI_1D(monomial_1D_DCI):
    lam, q_lam, pi_obs = monomial_1D_DCI(5)
    d_prob = CB.DCIProblem((lam, q_lam), pi_obs)

    d_prob.solve()
    e_r = d_prob.result['e_r'].values[0]
    kl = d_prob.result['kl'].values[0]

    assert np.round(e_r, 1) == 0.8
    assert np.round(kl, 1) == 2.1
    assert isinstance(d_prob.dists['pi_in'], gkde)

def test_DCI_1D_density_plot(monomial_1D_DCI):
    lam, q_lam, pi_obs = monomial_1D_DCI(5)
    d_prob = CB.DCIProblem((lam, q_lam), pi_obs)

    d_prob.solve()
    e_r = d_prob.result['e_r'].values[0]
    kl = d_prob.result['kl'].values[0]
    axs = d_prob.density_plots()

    # TODO: Check title and more?
    assert len(axs) == 2

def test_DCI_1D_pi_in(monomial_1D_DCI):
    lam, q_lam, pi_obs = monomial_1D_DCI(5)
    d_prob = CB.DCIProblem((lam, q_lam), pi_obs, pi_in=uniform(-1, 2))

    d_prob.solve()
    e_r = d_prob.result['e_r'].values[0]
    kl = d_prob.result['kl'].values[0]

    assert np.round(e_r, 1) == 0.8
    assert np.round(kl, 1) == 2.1
    assert not isinstance(d_prob.dists['pi_in'], gkde)

def test_DCI_1D_pi_pr(monomial_1D_DCI):
    lam, q_lam, pi_obs = monomial_1D_DCI(1)
    d_prob = CB.DCIProblem((lam, q_lam), pi_obs,
                           pi_in=uniform(-1, 2),
                           pi_pr=uniform(-1, 2))

    d_prob.solve()
    e_r = d_prob.result['e_r'].values[0]
    kl = d_prob.result['kl'].values[0]

    assert np.round(e_r, 1) == 1.0
    assert np.round(kl, 1) == 1.6
    assert not isinstance(d_prob.dists['pi_in'], gkde)
    assert not isinstance(d_prob.dists['pi_pr'], gkde)

def test_DCI_1D_df(monomial_1D_DCI_df):
    samples, pi_obs = monomial_1D_DCI_df(5)
    d_prob = CB.DCIProblem(samples, pi_obs)

    d_prob.solve()
    e_r = d_prob.result['e_r'].values[0]
    kl = d_prob.result['kl'].values[0]

    assert np.round(e_r, 1) == 0.8
    assert np.round(kl, 1) == 2.1
    assert isinstance(d_prob.dists['pi_in'], gkde)

# def test_MUD_1D(monomial_1D_MUD):
#     lam, q_lam, data, std_dev = monomial_1D_MUD
#     d_prob = CB.MUDProblem((lam, q_lam), data, std_dev)
# 
#     d_prob.solve()
#     e_r = d_prob.result['e_r'].values[0]
#     kl = d_prob.result['kl'].values[0]
# 
#     assert np.round(e_r, 1) == 0.8
#     assert np.round(kl, 1) == 2.1
# 
# 
#  def test_identity_1D_mud(identity_problem_mud_1D):
#      # Arrange
#      D = identity_problem_mud_1D
#  
#      # Act
#      mud_point = D.estimate()
#      ratio = D._r
#  
#      # Assert
#      assert 1 == D.n_params
#      assert 1 == D.n_features
#      assert 2000 == D.n_samples
#      assert np.round(mud_point, 1) == 0.5
#      assert np.abs(np.mean(ratio) - 1) < 0.2
#  
#  
#  def test_identity_mud_1D_with_equal_weights(identity_problem_mud_1D_equal_weights):
#      # Arrange
#      D = identity_problem_mud_1D_equal_weights
#  
#      # Act
#      mud_point = D.estimate()
#      ratio = D._r
#  
#      # Assert
#      assert np.round(mud_point, 1) == 0.5
#      assert np.abs(np.mean(ratio) - 1) < 0.2
#  
#  
#  def test_identity_mud_1D_with_biased_weights(identity_problem_mud_1D_bias_weights):
#      # Arrange
#      D = identity_problem_mud_1D_bias_weights
#  
#      # Act
#      mud_point = D.estimate()
#      updated_density = D._up
#      ratio = D._r
#  
#      # Assert
#      assert np.round(mud_point, 1) == 0.5
#      assert np.sum(updated_density) > 0
#      assert np.mean(ratio) > 0
#  
#  
#  def test_identity_mud_2D_with_normalized_weights(identity_problem_mud_1D_equal_weights):
#      # Arrange
#      D = identity_problem_mud_1D_equal_weights
#  
#      # Act
#      D.set_weights(np.ones(D.n_samples), normalize=True)
#  
#      # Assert
#      assert (np.sum(D._weights) - 70.710) < 0.01
#  
#  
#  def test_identity_mud_problem_1D_plot_params(
#      identity_problem_mud_1D_domain, identity_problem_mud_1D_equal_weights
#  ):
#      # Arrange
#      D_base = identity_problem_mud_1D_domain
#      D_weights = identity_problem_mud_1D_equal_weights
#  
#      # Act
#      mud_point_base = D_base.estimate()
#      mud_point_weights = D_weights.estimate()
#      ratio_base = D_base._r
#      ratio_weights = D_weights._r
#      ax1 = D_base.plot_param_space(true_val=[0.5])
#      ax2 = D_weights.plot_param_space(
#          win_opts={}, in_opts=None, up_opts=None, mud_opts=None
#      )
#  
#      # Assert
#      assert np.round(mud_point_base, 1) == 0.5
#      assert np.abs(np.mean(ratio_base) - 1) < 0.5
#      assert np.round(mud_point_weights, 1) == 0.5
#      assert np.abs(np.mean(ratio_weights) - 1) < 0.2
#      assert len(ax1.get_lines()) == 4
#      for line in ax1.get_lines():
#          if "init" in line.get_label():
#              # uniform initial, all same values
#              assert 1 == int(list(set(line.get_ydata()))[0])
#          if "update" in line.get_label():
#              # peak of dist should be around the center of [0, 1] domain
#              assert np.abs(0.5 - line.get_xdata()[np.argmax(line.get_ydata())]) < 0.1
#          if "MUD" in line.get_label():
#              # Mud value should be verticle line (same x) around 0.5
#              assert np.abs(line.get_xdata()[0] - 0.5) < 0.1
#          if "dagger" in line.get_label():
#              assert np.abs(line.get_xdata()[0] - 0.5) < 0.0001
#      assert len(ax2.get_lines()) == 1
#      assert "tilde" in ax2.get_lines()[0].get_label()
#      assert np.abs(np.mean(ax2.get_lines()[0].get_ydata()) - 1) < 0.25
#  
#  
#  def test_identity_mud_problem_1D_plot_obs(identity_problem_mud_1D_domain):
#      # Arrange
#      D_base = identity_problem_mud_1D_domain
#  
#      # Act
#      mud_point_base = D_base.estimate()
#      ratio_base = D_base._r
#      ax1 = D_base.plot_obs_space()
#      ax2 = D_base.plot_obs_space(ob_opts=None, pr_opts=None, pf_opts=None)
#      average_diff = np.average(
#          np.abs(ax1.lines[2].get_ydata() - ax1.lines[0].get_ydata())
#      )
#  
#      # Assert
#      assert np.round(mud_point_base, 1) == 0.5
#      assert np.abs(np.mean(ratio_base) - 1) < 0.2
#      assert len(ax1.get_lines()) == 3
#      assert average_diff < 0.1
#      for line in ax1.get_lines():
#          if "obs" in line.get_label():
#              # Observed distribution should be normal peaked around true val 0.5
#              assert np.abs(0.5 - line.get_xdata()[np.argmax(line.get_ydata())]) < 0.1
#          if "pred" in line.get_label():
#              # Predicted should be same as initial since identity map
#              vals = list(set(line.get_ydata()))
#              vals.sort()
#              assert 0 == vals[0]
#              assert 0.01 > (1 - vals[-1])
#          if "pf-pr" in line.get_label():
#              assert line.get_xdata()[np.argmax(line.get_ydata())]
#      assert len(ax2.get_lines()) == 0
#  
#  
#  def test_identity_map_problem_1D(identity_problem_map_1D):
#      # Arrange
#      D = identity_problem_map_1D
#  
#      # Act
#      map_point = D.estimate()
#  
#      # Assert
#      assert np.round(map_point, 1) == 0.5
