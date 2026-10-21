# Cut-off and credit-limit strategy (out-of-time test set)

Applying a PD cut-off of 0.12 (chosen on validation) to the out-of-time test set changes approvals by -36 pts, cuts the default rate among approved loans from 15.8% to 10.2% and moves net profit from -40,954 to -2,946 ETB.

Economics: fee on repaid loans, LGD 85% on defaults, cost of funds 1% of principal. Cut-off chosen on the validation set only.

## Cut-off comparison (test)

| Policy | Approval | Default rate (approved) | Net profit (ETB) |
|---|---|---|---|
| Approve all | 100% | 15.8% | -40,954 |
| PD <= 0.12 (policy) | 64% | 10.2% | -2,946 |
| PD <= 0.09 (tighter) | 51% | 8.9% | 6,468 |

## Limit tiers (approved loans, test)

| Tier | Loans | Mean PD | Default rate | Limit multiplier | Profit flat (ETB) | Profit tiered (ETB) |
|---|---|---|---|---|---|---|
| Tier 1 (lowest risk) | 828 | 3.5% | 4.8% | x1.5 | 12,864 | 19,297 |
| Tier 2 | 1072 | 5.9% | 9.8% | x1.0 | -1,152 | -1,152 |
| Tier 3 | 1140 | 9.3% | 14.6% | x0.6 | -14,658 | -8,795 |
