//> using scala 2.13.18
//> using dep org.apache.spark::spark-core:4.3.0
//> using jvm system
//> using javaOpt --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED

package bdalab.assignment1

import org.apache.spark.{SparkConf, SparkContext}

// From the repository root: scala-cli run "Assignment 1/StudentPerformanceAnalysis.scala"

case class Student(
    studentId: Int,
    name: String,
    department: String,
    semester: Int,
    marks: Double
)

object StudentPerformanceAnalysis {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("Student Academic Performance Analysis")
      .setIfMissing("spark.master", "local[2]")
    val sc = new SparkContext(conf)
    try {
      sc.setLogLevel("ERROR")
      val records = List(
        Student(101, "Aarav Sharma", "AI & DS", 5, 88.5),
        Student(102, "Priya Patel", "CSE", 5, 72.0),
        Student(103, "Rohan Verma", "AI & DS", 5, 94.0),
        Student(104, "Sneha Nair", "ECE", 5, 68.5),
        Student(105, "Vikram Singh", "CSE", 5, 81.0),
        Student(106, "Ananya Iyer", "AI & DS", 5, 76.5),
        Student(107, "Karthik Raj", "ECE", 5, 85.0),
        Student(108, "Meera Krishnan", "CSE", 5, 64.0),
        Student(109, "Devansh Gupta", "AI & DS", 5, 91.5),
        Student(110, "Divya Menon", "ECE", 5, 74.0)
      )
      val students = sc.parallelize(records, 3)

      println("ID   Name             Department  Semester  Marks")
      students.collect().foreach { s =>
        println(f"${s.studentId}%-4d ${s.name}%-16s ${s.department}%-11s ${s.semester}%8d ${s.marks}%6.1f")
      }

      // Count how many records have actually reached the filter.
      val checked = sc.longAccumulator("Records checked")
      val highNames = students.filter { s =>
        checked.add(1)
        s.marks > 75
      }.map(_.name)

      println(s"\nRecords checked before collect: ${checked.value}")
      val names = highNames.collect()
      println(s"Records checked after collect: ${checked.value}")
      println(s"Students above 75 (${names.length}): ${names.mkString(", ")}")

      val departmentCounts = students.map(s => (s.department, 1)).reduceByKey(_ + _)
      println("\nStudents per department")
      departmentCounts.collect().sortBy(_._1).foreach { case (department, count) =>
        println(s"$department: $count")
      }

      val average = students.map(_.marks).mean()
      val top = students.max()(Ordering.by[Student, Double](_.marks))
      println(f"\nAverage marks: $average%.2f")
      println(f"Highest marks: ${top.name}, ID ${top.studentId}, ${top.department}, ${top.marks}%.1f")

      println("\nRDD lineage for students above 75")
      println(highNames.toDebugString)
      println("\nRDD lineage for department counts")
      println(departmentCounts.toDebugString)
    } finally {
      sc.stop()
    }
  }
}
