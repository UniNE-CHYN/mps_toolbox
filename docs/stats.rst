Statistics module
=================

Provides histogram (proportion) and variogram (indicator variogram) functions.

Indicator variogram
-------------------

The variogram function is defined as follows:

.. math::
    2 \gamma_i ( \mathbf{h}) = \mathbb{E} [I_i(\mathbf{x}+\mathbf{h}) - I_i(\mathbf{x})]^2

where :math:`I_i(\mathbf{x})` is indicator function for category :math:`s_i`
