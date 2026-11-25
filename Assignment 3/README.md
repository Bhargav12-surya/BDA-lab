# Diabetes prediction with Spark MLlib

For this assignment I compared logistic regression, a decision tree and a random forest on 768 records from the Pima Indians Diabetes Dataset. The task was to predict the recorded diabetes outcome from eight measurements.

## Run the program

Open a terminal in the folder containing `DiabetesPrediction.scala` and `diabetes.csv`. Run the following command with JDK 21 and Scala CLI installed.

```sh
scala-cli run DiabetesPrediction.scala
```

The Scala file declares the Spark dependencies. The program prints the missing-value counts, model comparison and predictions for three fictional examples. The saved run is in [Spark output.txt](Spark%20output.txt).

## Data and results

I added a header to the [original CSV](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv). In this dataset, a pregnancy count of zero is valid. I treated zero glucose, blood pressure, skin thickness, insulin and BMI readings as missing. Insulin had the most missing readings, 374 out of 768. I used the training records to calculate the replacement medians and fit the scaler.

The split gave 447 training records, 150 validation records and 171 test records. I chose the random forest because it had the highest validation F1 score (0.6875). These are the results on the test records, with diabetes as the positive class.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.7602 | 0.7872 | 0.5441 | 0.6435 |
| Decision tree | 0.7368 | 0.6949 | 0.6029 | 0.6457 |
| Random forest | 0.7661 | 0.7414 | 0.6324 | 0.6825 |

The random forest identified 43 of the 68 positive test records and missed 25. It also predicted a positive outcome for 15 of the 103 negative records.
