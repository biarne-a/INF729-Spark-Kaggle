package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SparkSession}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession.builder.config(conf).appName("TP_spark").getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/
   val df: DataFrame = spark
     .read
     .option("inferSchema", "true") // Try to infer the data types of each column
     .parquet("data/prepared_trainingset")

    println(s"Total number of rows: ${df.count}")

    /** TF-IDF **/

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val wordsRemover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("cleaned_tokens")

    val countsVec = new CountVectorizer()
      .setInputCol("cleaned_tokens")
      .setOutputCol("raw_text_features")

    val idf = new IDF()
      .setInputCol("raw_text_features")
      .setOutputCol("tfidf")

    /** Conversion des données catégorielles **/
    var country2Indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    var currency2Indexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")


    /** VECTOR ASSEMBLER **/
    val vecAssembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")

    /** MODEL **/
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /** PIPELINE **/
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, wordsRemover, countsVec, idf, country2Indexer, currency2Indexer, vecAssembler, lr))

    /** TRAINING AND GRID-SEARCH **/
    val Array(trainDf, testDf) = df.randomSplit(Array[Double](0.9, 0.1))

    val paramGrid = new ParamGridBuilder()
      .addGrid(countsVec.minDF, Array(55.0, 75.0, 95.0))
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val model = trainValidationSplit.fit(trainDf)
    val df_WithPredictions = model.transform(testDf)
    val f1score = evaluator.evaluate(df_WithPredictions)

    println("F1-Score = " + f1score)

    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    model.write.overwrite.save("best_logisitic_regression")
  }
}
