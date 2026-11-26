# Vendored Social-GAN components

`models.py` and `losses.py` are copied from the Social-GAN implementation
distributed in the official RiskBench baseline repository. The accompanying
MIT license is retained in this directory.

The only code changes are device-compatibility changes: tensors that were
created with hard-coded `.cuda()` calls are now created on the same device as
their input tensors. The network architecture, forward operations, and loss
definitions are unchanged.

This code is used to retrain the published Social-GAN architecture on the
current family-disjoint safe-core split. It is not presented as an evaluation
of the original pretrained Social-GAN checkpoint.
