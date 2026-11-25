//> using scala 2.13.18
//> using dep org.apache.spark::spark-sql:4.3.0
//> using jvm system
//> using javaOpt --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED

package bdalab.assignment2

import java.nio.file.{Files, Paths}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

// From the repository root: scala-cli run "Assignment 2/SalesAnalysis.scala"
// To use another CSV or keep the Spark UI open, append: -- path/to/sales.csv --keep-ui
object SalesAnalysis {
  def main(args: Array[String]): Unit = {
    val file = args.find(_ != "--keep-ui").getOrElse {
      if (Files.isRegularFile(Paths.get("sales.csv"))) "sales.csv" else "Assignment 2/sales.csv"
    }
    val conf = new SparkConf().setAppName("Assignment 2 - Retail Sales")
      .setIfMissing("spark.master", "local[2]")
      .setIfMissing("spark.sql.shuffle.partitions", "2")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._

    try {
      sc.setLogLevel("ERROR")
      val fields = Seq("transaction_id", "product_name", "category", "store_location",
        "quantity_sold", "unit_price", "transaction_date")
      // Keep the original text so a bad price or date can be printed as entered.
      val schema = StructType(fields.map(StructField(_, StringType, nullable = true)))
      val raw = spark.read.schema(schema).option("header", "true")
        .option("enforceSchema", "false").option("ignoreLeadingWhiteSpace", "true")
        .option("ignoreTrailingWhiteSpace", "true").csv(file)

      println("\nCSV preview")
      raw.show(5, truncate = false)
      val parsed = raw
        .withColumn("qty", expr("try_cast(quantity_sold as int)"))
        .withColumn("price", expr("try_cast(unit_price as decimal(12,2))"))
        .withColumn("date", expr("try_cast(transaction_date as date)"))
      val required = Seq("transaction_id", "product_name", "category", "store_location")
      val checks = required.map(field =>
        when(length(trim(coalesce(col(field), lit("")))) === 0, lit(s"missing $field"))
      ) ++ Seq(
        when($"qty".isNull || $"qty" <= 0, "quantity must be a positive integer"),
        when($"price".isNull || $"price" <= 0 || !$"unit_price".rlike("^[0-9]+([.][0-9]{1,2})?$"),
          "price must be a positive amount with up to two decimals"),
        when($"date".isNull || !$"transaction_date".rlike("^[0-9]{4}-[0-9]{2}-[0-9]{2}$"),
          "invalid ISO date")
      )
      val checked = parsed.withColumn("issue", concat_ws(", ", checks: _*))
      println("\nRows with missing or invalid values")
      checked.filter($"issue" =!= "").select((fields.map(col) :+ $"issue"): _*)
        .orderBy("transaction_id").show(100, truncate = false)

      val sales = checked.filter($"issue" === "").select(
        $"transaction_id", $"product_name", $"category", $"store_location",
        $"qty".as("quantity_sold"), $"price".as("unit_price"), $"date".as("transaction_date")
      ).withColumn("total_sales", $"quantity_sold" * $"unit_price")

      println("\nSchema after conversion")
      sales.printSchema()
      val inputCount = raw.count()
      val validCount = sales.count()
      println(s"Input rows: $inputCount | Valid rows: $validCount | Excluded: ${inputCount - validCount}")

      println("\nSales amount per valid transaction (INR)")
      sales.orderBy("transaction_id").show(100, truncate = false)
      println("\nTotal revenue (INR)")
      sales.agg(sum("total_sales").as("revenue")).show(false)

      sc.setJobGroup("category-revenue", "Revenue by product category")
      val byCategory = sales.groupBy("category").agg(sum("total_sales").as("revenue"))
      val tracker = sc.statusTracker
      println(s"\nCategory jobs after groupBy and sum: ${tracker.getJobIdsForGroup("category-revenue").length}")
      val categoryRows = byCategory.collect()
      println(s"Category jobs after collect: ${tracker.getJobIdsForGroup("category-revenue").length}")
      sc.clearJobGroup()
      println("\nRevenue by category (INR)")
      println("Category                 Revenue")
      val categoryTotals = categoryRows.map(r => r.getString(0) -> BigDecimal(r.getDecimal(1)))
      categoryTotals.sortBy { case (category, revenue) => (-revenue, category) }.foreach {
        case (category, revenue) => println(f"$category%-19s $revenue%12.2f")
      }
      println("\nPhysical plan for category revenue")
      byCategory.explain("formatted")

      println("\nTop five products by units sold")
      sales.groupBy("product_name").agg(sum("quantity_sold").as("units_sold"))
        .orderBy(desc("units_sold"), asc("product_name")).limit(5).show(false)

      println("\nRevenue by store location (INR)")
      sales.groupBy("store_location").agg(sum("total_sales").as("revenue"))
        .orderBy(desc("revenue"), asc("store_location")).show(false)

      println("\nTransactions strictly above INR 10,000")
      sales.filter($"total_sales" > 10000)
        .orderBy(desc("total_sales"), asc("transaction_id")).show(100, truncate = false)

      println("\nCategory totals using an RDD")
      val rddTotals = sales.select("category", "total_sales").rdd
        .map(r => r.getString(0) -> BigDecimal(r.getDecimal(1)))
        .reduceByKey(_ + _)
      rddTotals.setName("Revenue by category using reduceByKey")
      val rddResult = rddTotals.collect().sortBy(_._1)
      rddResult.foreach { case (category, revenue) => println(f"$category%-19s $revenue%12.2f") }
      println(s"RDD and DataFrame totals match: ${rddResult.toMap == categoryTotals.toMap}")
      println(rddTotals.toDebugString)

      if (args.contains("--keep-ui")) {
        println(s"Spark UI: ${sc.uiWebUrl.getOrElse("disabled")}")
        println("Press Enter to finish.")
        scala.io.StdIn.readLine()
      }
    } finally {
      spark.stop()
    }
  }
}
