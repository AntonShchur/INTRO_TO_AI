ff_10_error = sum((abs(feed_forward_10_outputs - target)) ./ (target + 1)) * 100
ff_20_error = sum((abs(feed_forward_20_outputs - target)) ./ (target + 1)) * 100
cascade_20_error = sum((abs(cascade_20_outputs - target)) ./ (target + 1)) * 100
cascade_2_10_error = sum((abs(cascade_2_10_outputs - target)) ./ (target + 1))* 100
elman_15_error = sum((abs(elman_15_outputs - target)) ./ (target + 1)) * 100
elman_3_error = sum((abs(elman_3_5_outputs - target)) ./ (target + 1)) * 100