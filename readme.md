
# Implementation of SGd sequential & mini-batch SGD sequential form.

# Comparison of two different SGD sequential algo.

By implementing stochastic gradient descent, I evaluate the effects of
choosing different learning rates, batch sizes and sampling schemes on the error rate and runtime. Overall,
I discovered that (1) while the runtime of each algorithm varies, implementing different sampling schemes
will not have any significant impact on the error rate. (2) Choosing different learning rates will change both
the pattern of error rate and runtime. In general, given fixed batch size, as we increase the step size, the
performance of model is getting better with lower error rate. (3) I also implemented the algorithms with
different batch sizes. With a fixed learning rate, the smaller batch size generated a lower test error.
In the next three sections, we will further elaborate on these observations.
