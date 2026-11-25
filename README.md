# BDA lab

These are my three Big Data Analytics lab assignments. I used Scala and Apache Spark to work with student records, retail sales and a public diabetes dataset.

| Assignment | What I studied | Files |
| --- | --- | --- |
| [1. Student performance with RDDs](Assignment%201/) | Filtering marks, counting students by department and tracing when Spark starts a job | [Code](Assignment%201/StudentPerformanceAnalysis.scala) · [Student records](Assignment%201/students.csv) · [Output](Assignment%201/Spark%20output.txt) · [Report](Assignment%201/Assignment%201%20-%20Student%20Performance.docx) |
| [2. Retail sales with DataFrames](Assignment%202/) | Checking invalid transactions, comparing revenue across stores and tracing a category total through the Spark UI | [Code](Assignment%202/SalesAnalysis.scala) · [Sales data](Assignment%202/sales.csv) · [Output](Assignment%202/Spark%20output.txt) · [Report](Assignment%202/Assignment%202%20-%20Sales%20Analysis.docx) · [Spark UI screenshots](Assignment%202/Spark%20UI/) |
| [3. Diabetes prediction with MLlib](Assignment%203/) | Comparing logistic regression, a decision tree and a random forest on the same held-out records | [Code](Assignment%203/DiabetesPrediction.scala) · [Dataset](Assignment%203/diabetes.csv) · [Output](Assignment%203/Spark%20output.txt) · [Results and interpretation](Assignment%203/README.md) |

## Running the assignments

I used JDK 21, Scala CLI and Spark 4.3.0. Run one command at a time from the repository root. Scala CLI reads the dependency versions from each source file.

```sh
scala-cli run "Assignment 1/StudentPerformanceAnalysis.scala"
scala-cli run "Assignment 2/SalesAnalysis.scala"
scala-cli run "Assignment 3/DiabetesPrediction.scala"
```

To inspect the sales query yourself, run `scala-cli run "Assignment 2/SalesAnalysis.scala" -- --keep-ui`. Open the Spark UI address printed in the terminal. Press Enter when you finish.

## Datasets

I created the ten student records in the first program and included a CSV copy. I also created the retail sales CSV, including four invalid rows to test the cleaning step. For Assignment 3, I used the Pima Indians Diabetes Dataset and added a header to the CSV. [Original diabetes data](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) · [Field descriptions](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names)
