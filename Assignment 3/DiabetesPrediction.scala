//> using scala 2.13.18
//> using dep org.apache.spark::spark-mllib:4.3.0
//> using jvm system
//> using javaOpt --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED

package bdalab.assignment3

import java.nio.file.{Files, Paths}
import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.feature.{Imputer, SQLTransformer, StandardScaler, VectorAssembler}
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

// From the repository root: scala-cli run "Assignment 3/DiabetesPrediction.scala"
// Original CSV (a header was added to the local copy):
// https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
// Field definitions: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names
object DiabetesPrediction {
  val inputs = Array("Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age")
  val cleanCols = inputs.map(_ + "_clean")
  val filledCols = inputs.map(_ + "_filled")

  // Keep zero pregnancies. In this dataset zeros in five measurement fields stand for missing values.
  def cleaningStage(): SQLTransformer = {
    val expressions = inputs.map { name =>
      val domain = name match {
        case "Pregnancies" => s"$name >= 0 AND $name = floor($name)"
        case "Age" => s"$name > 0 AND $name = floor($name)"
        case "DiabetesPedigreeFunction" => s"$name >= 0"
        case _ => s"$name > 0"
      }
      s"CASE WHEN $domain AND abs($name) < double('Infinity') " +
        s"THEN $name ELSE CAST(NULL AS DOUBLE) END AS ${name}_clean"
    }
    new SQLTransformer().setStatement("SELECT *, " + expressions.mkString(", ") + " FROM __THIS__")
  }

  case class Scores(tp: Long, tn: Long, fp: Long, fn: Long) {
    private def ratio(a: Long, b: Long): Double = if (b == 0) 0.0 else a.toDouble / b
    def accuracy: Double = ratio(tp + tn, tp + tn + fp + fn)
    def precision: Double = ratio(tp, tp + fp)
    def recall: Double = ratio(tp, tp + fn)
    def f1: Double = ratio(2 * tp, 2 * tp + fp + fn)
  }

  def score(predictions: DataFrame): Scores = {
    val counts = predictions.groupBy("Outcome", "prediction").count().collect().map { r =>
      (r.getDouble(0), r.getDouble(1)) -> r.getLong(2)
    }.toMap.withDefaultValue(0L)
    Scores(counts((1, 1)), counts((0, 0)), counts((0, 1)), counts((1, 0)))
  }

  def printScores(heading: String, results: Seq[(String, Scores)]): Unit = {
    println("\n" + heading + " (positive class = 1)")
    println("Model                  Accuracy  Precision  Recall     F1    TP   TN   FP   FN")
    results.foreach { case (name, s) =>
      println(f"$name%-21s  ${s.accuracy}%8.4f  ${s.precision}%9.4f  ${s.recall}%6.4f  ${s.f1}%6.4f  ${s.tp}%3d  ${s.tn}%3d  ${s.fp}%3d  ${s.fn}%3d")
    }
  }

  def main(args: Array[String]): Unit = {
    val csv = args.headOption.getOrElse {
      if (Files.isRegularFile(Paths.get("diabetes.csv"))) "diabetes.csv" else "Assignment 3/diabetes.csv"
    }
    val conf = new SparkConf().setAppName("Assignment 3 - Diabetes prediction")
      .setIfMissing("spark.master", "local[2]").setIfMissing("spark.sql.shuffle.partitions", "2")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._
    try {
      spark.sparkContext.setLogLevel("ERROR")
      val schema = StructType((inputs :+ "Outcome").map(StructField(_, DoubleType, nullable = true)))
      val raw = spark.read.option("header", "true").option("enforceSchema", "false").schema(schema).csv(csv)
      val labelled = raw.filter($"Outcome".isin(0.0, 1.0))
      val total = raw.count()
      println(s"Dataset: $total records. Missing or invalid outcomes excluded: ${total - labelled.count()}")
      println("Inputs: " + inputs.mkString(", ") + ". Target: Outcome (0 = negative, 1 = positive).")
      val audit = cleaningStage().transform(labelled)
      val missing = inputs.map(n => count(when(col(n + "_clean").isNull, 1)).as(n))
      println("\nMissing or invalid feature values before median imputation")
      audit.agg(missing.head, missing.tail.toIndexedSeq: _*).show(false)

      // Split by the eight inputs only. Identical feature records always stay in the same group.
      val bucket = pmod(xxhash64(inputs.toIndexedSeq.map(col): _*), lit(100))
      val split = labelled.withColumn("split", when(bucket < 60, "train")
        .when(bucket < 80, "validation").otherwise("test")).cache()
      println("Actual split sizes and class counts (target proportions 60/20/20)")
      split.groupBy("split").agg(count(lit(1)).as("rows"),
        count(when($"Outcome" === 0, 1)).as("negative"), count(when($"Outcome" === 1, 1)).as("positive"))
        .orderBy("split").show(false)
      val train = split.filter($"split" === "train")
      val validation = split.filter($"split" === "validation")
      val test = split.filter($"split" === "test")

      println("I fitted the median imputer and the standard scaler on the training rows only. " +
        "The pipeline fills missing values, assembles the eight inputs into a vector, scales it " +
        "and passes it to the classifier.")
      val imputer = new Imputer().setInputCols(cleanCols).setOutputCols(filledCols).setStrategy("median")
      val assembler = new VectorAssembler().setInputCols(filledCols).setOutputCol("rawFeatures")
      val scaler = new StandardScaler().setInputCol("rawFeatures").setOutputCol("features")
        .setWithMean(true).setWithStd(true)
      val classifiers: Seq[(String, PipelineStage)] = Seq(
        "Logistic regression" -> new LogisticRegression().setLabelCol("Outcome").setMaxIter(100).setRegParam(0.01),
        "Decision tree" -> new DecisionTreeClassifier().setLabelCol("Outcome").setMaxDepth(4).setSeed(42L),
        "Random forest" -> new RandomForestClassifier().setLabelCol("Outcome").setNumTrees(60).setMaxDepth(5).setSeed(42L)
      )
      val models = classifiers.map { case (name, classifier) =>
        val stages = Array[PipelineStage](cleaningStage(), imputer, assembler, scaler, classifier)
        name -> new Pipeline().setStages(stages).fit(train)
      }
      val validationScores = models.map { case (name, model) => name -> score(model.transform(validation)) }
      printScores("Validation comparison", validationScores)
      val (best, validScore) = validationScores.maxBy { case (_, s) => (s.f1, s.recall) }
      println(f"Selected model: $best (validation F1 ${validScore.f1}%.4f, with recall as the tie-breaker).")
      val testScores = models.map { case (name, model) => name -> score(model.transform(test)) }
      printScores("Held-out test comparison", testScores)
      val bestTest = testScores.maxBy { case (_, s) => (s.f1, s.recall) }._1
      println(s"Highest test F1 on this split: $bestTest.")
      val s = testScores.toMap.apply(best)
      println(s"On the test set, $best identified ${s.tp} of ${s.tp + s.fn} positive records " +
        s"and marked ${s.fp} of ${s.tn + s.fp} negative records as positive.")

      // Fictional examples use the same feature order and units as the historical records.
      val examples = Seq(
        ("Example A", 1.0, 88.0, 66.0, 22.0, 70.0, 23.5, 0.25, 25.0),
        ("Example B", 6.0, 174.0, 86.0, 36.0, 150.0, 38.2, 0.70, 51.0),
        ("Example C", 2.0, 124.0, 74.0, 28.0, 0.0, 30.1, 0.44, 34.0)
      ).toDF((Array("id") ++ inputs).toIndexedSeq: _*)
      val selectedModel = models.toMap.apply(best)
      println(s"\nFictional new records scored by $best (Example C has missing insulin)")
      selectedModel.transform(examples)
        .withColumn("positive_score", round(vector_to_array($"probability")(1), 4))
        .withColumn("predicted_class", when($"prediction" === 1, "positive").otherwise("negative"))
        .select((Array("id") ++ inputs ++ Array("positive_score", "predicted_class")).toIndexedSeq.map(col): _*)
        .orderBy("id").show(false)
      println("The score is the model output for class 1. The original dataset covers adult women " +
        "of Pima heritage, so these results describe performance within that sample.")
      println("\nHow Spark shares the work")
      println("Spark runs transformations on partitions. Logistic regression combines gradient summaries " +
        "from the partitions. Tree models combine statistics to choose splits. The driver coordinates " +
        "training, and executors can score different rows in parallel.")
      println(s"Spark master for this run: ${spark.sparkContext.master}")
      split.unpersist()
    } finally spark.stop()
  }
}
