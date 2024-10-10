---
layout: post
title:  fe exotic option
date:   2024-10-09 11:24:29 +0800
categories: 
    - financial 
    - python
---

<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

可以研究`exotic option`的相关的内容.

1. **Barrier Options**: Options that are activated or extinguished when the underlying asset reaches a certain price level (the barrier).
   - **Knock-In Options**: Become active only if the underlying asset hits the barrier.
   - **Knock-Out Options**: Become inactive if the underlying asset hits the barrier.

2. **Asian Options**: The payoff depends on the average price of the underlying asset over a certain period rather than the price at maturity.

3. **Lookback Options**: Allow the holder to "look back" over time to determine the optimal exercise price based on the underlying asset's minimum or maximum price during the option's life.

4. **Digital (Binary) Options**: Provide a fixed payoff if the underlying asset meets certain conditions at expiration.

5. **Chooser Options**: Allow the holder to choose at a certain point in time whether the option is a call or a put.

6. **Rainbow Options**: Options on multiple underlying assets, where the payoff depends on the performance of two or more assets.

7. **Spread Options**: Options where the payoff depends on the difference (spread) between the prices of two underlying assets.
