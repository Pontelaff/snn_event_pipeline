# Layer Comparision

This document contains tests and results comparing the layer behaviour and neuron activity of this implementation to the pytorch implementation. The input sequences used are taken from the pytorch implementation to make for an easier comparison where for each new event window the timestamp is increased by 100us

## Head Layer

(channel, x_pos, y_pos) = (1, 1, 1)

channel mean error = 0.087121

layer mean error = 0.054806

```python
LEAK_RATE = 0.17 * LOG_BINSIZE
U_RESET = 0
U_THRESH = 0.74
REC_DELAY = 100
REFRACTORY_PERIOD = 50
```

![Head Layer Activity](head_out_1_1_1.png)


## G1 Layer

(channel, x_pos, y_pos) = (12, 1, 1)

channel mean error = 0.056391

layer mean error = 0.150023

```python
LEAK_RATE = 0.17 * LOG_BINSIZE
U_RESET = 0
U_THRESH = 1.5
REC_DELAY = 100
REFRACTORY_PERIOD = 50
```

![G1 Layer Activity](G1_out_18_1_1.png)

## R1a Layer

(channel, x_pos, y_pos) = (12, 1, 1)

channel mean error = 0.556818

layer mean error = 0.333215

```python
LEAK_RATE = 0.17 * LOG_BINSIZE
U_RESET = 0
U_THRESH = 1.0
REC_DELAY = 100
REFRACTORY_PERIOD = 50
```

![R1a Layer Activity](R1a_out_12_1_1.png)