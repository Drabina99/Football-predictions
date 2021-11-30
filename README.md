# Football predictions

The idea of the project is to compare results of 4 predictive models:

- linear regression
- GRNN
- MLP
- RBFNN

The data used to verify the models' skills is a CSV file containg some information about Manchester United matches in English Premier League (2017-2021). Variables which are taken into consideration are:

- round number
- sum of points from the recent 5 matches (United)
- sum of point from the recent 5 matches (opponent)
- goal difference from the recent 5 matches (United)
- location of the game (home/away)
- last season position of the opponent
  (value 18 for the winner of Championship, 19 for the runner-up and 20 for the last newcomer)
  
By default there are 10 iterations. The data is shuffled in each of it. Then, models are trained on some training data and verified on test data.
After each iteration predictions can be seen in a form of a table:

| GAME                       | PRED_REGR   | PRED_GRNN   | PRED_MLP   | PRED_RBF   | REAL      |          LR |       GRNN |   MLP |   RBF |
|----------------------------+-------------+-------------+------------+------------+-----------+-------------+------------+-------+-------|
| Brighton 2:3 Man Utd       | draw        | draw        | win        | win        | win (1)   |  0.222144   |  0.174716  |     1 |     1 |
| Man Utd 2:1 West Ham       | win         | win         | win        | win        | win (1)   |  0.391575   |  0.574699  |     1 |     1 |
| Leicester 2:2 Man Utd      | draw        | draw        | draw       | draw       | draw (0)  |  0.0542485  | -0.0872624 |     0 |     0 |
| Man Utd 3:0 Watford        | win         | win         | win        | win        | win (1)   |  0.543462   |  0.384777  |     1 |     1 |
| Spurs 0:1 Man Utd          | draw        | draw        | draw       | win        | win (1)   |  0.110582   | -0.124844  |     0 |     1 |
| Watford 2:0 Man Utd        | win         | win         | win        | win        | loss (-1) |  0.351904   |  0.362033  |     1 |     1 |
| West Ham 1:3 Man Utd       | draw        | draw        | win        | win        | win (1)   |  0.25241    |  0.270066  |     1 |     1 |
| Wolves 2:1 Man Utd         | win         | win         | win        | win        | loss (-1) |  0.360259   |  0.579989  |     1 |     1 |
| Man Utd 1:2 Crystal Palace | win         | win         | win        | win        | loss (-1) |  0.437339   |  0.420657  |     1 |     1 |
| Man Utd 1:0 Brighton       | win         | win         | win        | win        | win (1)   |  0.540189   |  0.431614  |     1 |     1 |
| Spurs 1:1 Man Utd          | draw        | win         | win        | win        | draw (0)  |  0.317309   |  0.428962  |     1 |     1 |
| Bournemouth 1:0 Man Utd    | win         | win         | win        | win        | loss (-1) |  0.415932   |  0.404468  |     1 |     1 |
| Arsenal 0:0 Man Utd        | draw        | draw        | win        | win        | draw (0)  |  0.0672328  |  0.0775775 |     1 |     1 |
| Chelsea 2:2 Man Utd        | draw        | draw        | draw       | draw       | draw (0)  | -0.0399517  | -0.0639867 |     0 |     0 |
| Man Utd 4:1 Newcastle      | draw        | draw        | win        | win        | win (1)   |  0.284136   |  0.197068  |     1 |     1 |
| Crystal Palace 2:3 Man Utd | win         | win         | loss       | win        | win (1)   |  0.377016   |  0.346937  |    -1 |     1 |
| Southampton 1:1 Man Utd    | win         | win         | win        | win        | draw (0)  |  0.440347   |  0.564307  |     1 |     1 |
| Man Utd 4:1 Fulham         | win         | win         | win        | win        | win (1)   |  0.595473   |  0.421195  |     1 |     1 |
| Man Utd 1:1 West Ham       | win         | draw        | draw       | loss       | draw (0)  |  0.470606   |  0.123243  |     0 |    -1 |
| Newcastle 1:4 Man Utd      | draw        | win         | win        | win        | win (1)   |  0.0746492  |  0.438614  |     1 |     1 |
| Man Utd 0:0 Southampton    | win         | win         | win        | win        | draw (0)  |  0.386192   |  0.413738  |     1 |     1 |
| Man Utd 2:2 Burnley        | draw        | draw        | draw       | win        | draw (0)  |  0.217025   |  0.0564604 |     0 |     1 |
| Man Utd 0:0 Man City       | draw        | draw        | draw       | draw       | draw (0)  | -0.00159574 | -0.102191  |     0 |     0 |
| Chelsea 0:2 Man Utd        | draw        | draw        | win        | win        | win (1)   |  0.116646   |  0.259289  |     1 |     1 |
| Man Utd 5:2 Bournemouth    | win         | win         | win        | win        | win (1)   |  0.700223   |  0.332818  |     1 |     1 |
| Man Utd 2:0 Huddersfield   | win         | win         | win        | win        | win (1)   |  0.637608   |  0.421232  |     1 |     1 |
| Man Utd 0:0 Wolves         | win         | win         | win        | win        | draw (0)  |  0.359207   |  0.578564  |     1 |     1 |
| Man Utd 2:1 Brighton       | win         | win         | draw       | win        | win (1)   |  0.541223   |  0.472905  |     0 |     1 |
| West Brom 1:1 Man Utd      | win         | draw        | win        | win        | draw (0)  |  0.848824   |  0.274934  |     1 |     1 |
| Man City 1:2 Man Utd       | draw        | draw        | draw       | draw       | win (1)   |  0.0380911  |  0.227664  |     0 |     0 |

Basically, value 1 means United's win, 0 is a draw and -1 is a win of the opponent. Linear regression and GRNN return float values, so it has to be classified that way:

- -1 to -0.33 - opponent's win
- -0.33 to 0.33 - draw
- 0.33 to 1 - United's win

After all iterations, a summary is printed:

| MODEL             |   CORRECT (/30) |   ACCURACY (%) |
|-------------------+-----------------+----------------|
| LINEAR REGRESSION |             134 |        44.6667 |
| GRNN              |             136 |        45.3333 |
| MLP               |             149 |        49.6667 |
| RBFNN             |             154 |        51.3333 |

RBFNN source: https://towardsdatascience.com/most-effective-way-to-implement-radial-basis-function-neural-network-for-classification-problem-33c467803319
