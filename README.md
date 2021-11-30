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

<img width="847" alt="a" src="https://user-images.githubusercontent.com/72979673/144042364-f28e403e-c670-4b6f-a7ab-b15b86786a76.png">

Basically, value 1 means United's win, 0 is a draw and -1 is a win of the opponent. Linear regression and GRNN return float values, so it has to be classified that way:

- -1 to -0.33 - opponent's win
- -0.33 to 0.33 - draw
- 0.33 to 1 - United's win

After all iterations, a summary is printed:

<img width="356" alt="b" src="https://user-images.githubusercontent.com/72979673/144043322-3b044160-c5d3-4de4-952f-5777775a05b4.png">

RBFNN source: https://towardsdatascience.com/most-effective-way-to-implement-radial-basis-function-neural-network-for-classification-problem-33c467803319
