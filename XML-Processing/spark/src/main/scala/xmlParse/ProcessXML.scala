package xmlParse

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import com.databricks.spark.xml
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.DataFrame

/**
 * @author agohil
 */
object ProcessXML {

  def main(args : Array[String]) {
    val conf = new SparkConf()
      .setAppName("ProcessXML")
      .setMaster("local[2]")

    val sc = new SparkContext(conf)
    val sqlContext : SQLContext = new org.apache.spark.sql.SQLContext(sc)
    
    //Load pages from XML file
    loadPages(sqlContext)
    
   /* println("orig count = " + colCount)
    println("sampled count = " + smpCount)*/
  }
  
  def loadPages(sqlContext : SQLContext) = {
    var df : DataFrame = null
    var newDF : DataFrame = null
    
    import sqlContext.implicits._
    
    df = sqlContext.read
         .format("xml")
         .option("rowTag", "page")
         .load("sample.xml")
         
    df.printSchema()
         
  }

}
