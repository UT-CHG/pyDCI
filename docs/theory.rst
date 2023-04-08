======
Theory
======

Data-Consistent Inversion comprises a set of methods to solve inverse problems founded in rigorous measure theory.

Notation
========

- :math:`\lambda \in \Lambda \subset \mathbb{R}^n` - parameter(s) of
interest
- :math:`Q(\lambda):\mathbb{R}^n \rightarrow \mathbb{R}^m` parameter to
observable <font color='red'>Quantity of Interest (QoI) Map</font>.
- :math:`d \in D := Q(\Lambda) \subset \mathbb{R}^m` - observable(s)

Inverse Problems
================

An inverse problem is posed when one wants to determine a set of parameters that produced observed data. A variety of typed of inverse problems can be posed, depending on the assumptions on the nature of uncertainty in the model parameters and outputs. 

Data-Consistent Solution
========================

Solution is given by a form of Bayes's rule that incorporates the push-forward
of a prior distribution , :math:`\pi^{pred}_\mathcal{D}`:

.. math::

    \pi^{up}_\Lambda(\lambda) = \pi^{in}_\Lambda(\lambda)\frac{\pi^{obs}_\mathcal{D}(Q(\lambda))}{\pi^{pred}_\mathcal{D}(Q(\lambda))}

Fore more information on theoretical formulations. See references below.

References
==========

[1] T. Butler, J. Jakeman, and T. Wildey, “Combining Push-Forward Measures
and Bayes’ Rule to Construct Consistent Solutions to Stochastic Inverse
Problems,” SIAM J. Sci. Comput., vol. 40, no. 2, pp. A984–A1011, Jan. 2018,
doi: 10.1137/16M1087229.
