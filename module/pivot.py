from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SQLContext
from pyspark import SparkContext as sc
from pyspark.sql.types import StructType
from pyspark.sql.functions import *
from functools import reduce

class Pivot:
    #df = SQLContext.createDataFrame(sc.emptyRDD(), StructType([]))
    def __init__(self, config, sparkConf, pivotFuture = None, pivotPast = None):
        self.conf = config
        self.sparkConf = sparkConf
        self.pivotFuture = pivotFuture
        self.pivotPast = pivotPast
        if self.conf.sourceCategory == 'all':
            self.phylomemyRoot = self.conf.dataPrefix + self.conf.storageDir + self.conf.algorithm + '/' + self.conf.source + '/' + self.conf.corpusName + '/'+ str(self.conf.startYear) + str(self.conf.endYear) +str(self.conf.intervalYear) + str(self.conf.stepYear) + '/' + str(self.conf.vocabSize) + '_' + str(self.conf.topicNumber) + '_' + str(self.conf.maxIterations) + '_' + str(self.conf.maxTermNbrPerTopic) + '/'
        else:  
            self.phylomemyRoot = self.conf.dataPrefix + self.conf.storageDir + self.conf.algorithm + '/' + self.conf.source + '/' + self.conf.corpusName + '/'+ self.conf.sourceCategory + '/' + str(self.conf.startYear) + str(self.conf.endYear) +str(self.conf.intervalYear) + str(self.conf.stepYear) + '/' + str(self.conf.vocabSize) + '_' + str(self.conf.topicNumber) + '_' + str(self.conf.maxIterations) + '_' + str(self.conf.maxTermNbrPerTopic) + '/'
        
        if self.conf.termlistAddr != '':
            self.topicNodesFile = self.phylomemyRoot + 'termList/topicNodes'
            self.similarityLinksFile = self.phylomemyRoot + 'termList/similarityLinks'
            self.topicDictionaryFile = self.phylomemyRoot + 'termList/topicDictionary'
            self.evolutionPathAddr = self.phylomemyRoot + 'termList/evolutionPath/'
            self.statisticsFile = self.phylomemyRoot + 'termList/statistics'
        else: 
            self.topicNodesFile = self.phylomemyRoot + 'corenlp/topicNodes'
            self.similarityLinksFile = self.phylomemyRoot + 'corenlp/similarityLinks'
            self.topicDictionaryFile = self.phylomemyRoot + 'corenlp/topicDictionary'
            self.evolutionPathAddr = self.phylomemyRoot + 'corenlp/evolutionPath/'
            self.statisticsFile = self.phylomemyRoot + 'corenlp/statistics'
            
    
    def initialize(self):
        future = self.sparkConf.spark.read.json(self.statisticsFile+'_future')
        past = self.sparkConf.spark.read.json(self.statisticsFile+'_past')
        return Pivot(self.conf, self.sparkConf, future, past)
    
#     def condition(self, cdt):
#         return self.pivot.where(cdt)
    
    #### Term filters
    def conditionArrayContains(self, df, attr, kw):
        return array_contains(df[attr], kw)
    
    def emerge(self, kw):
        future = None
        past = None
        if self.pivotFuture != None:
            future = self.pivotFuture.where(self.conditionArrayContains(self.pivotFuture, 'emerging', kw))
        if self.pivotPast != None:
            past = self.pivotPast.where(self.conditionArrayContains(self.pivotPast, 'emerging', kw))     
        return Pivot(self.conf, self.sparkConf, future, past)
    
    def decay(self, kw):
        future = None
        past = None
        if self.pivotFuture != None:
            future = self.pivotFuture.where(self.conditionArrayContains(self.pivotFuture, 'decaying', kw))
        if self.pivotPast != None:
            past = self.pivotPast.where(self.conditionArrayContains(self.pivotPast, 'decaying', kw))     
        return Pivot(self.conf, self.sparkConf, future, past)
    
    def stable(self, kw):
        future = None
        past = None
        if self.pivotFuture != None:
            future = self.pivotFuture.where(self.conditionArrayContains(self.pivotFuture, 'stable', kw))
        if self.pivotPast != None:
            past = self.pivotPast.where(self.conditionArrayContains(self.pivotPast, 'stable', kw))     
        return Pivot(self.conf, self.sparkConf, future, past)
    
    def specific(self, kw):
        future = None
        past = None
        if self.pivotFuture != None:
            future = self.pivotFuture.where(self.conditionArrayContains(self.pivotFuture, 'specific', kw))
        if self.pivotPast != None:
            past = self.pivotPast.where(self.conditionArrayContains(self.pivotPast, 'specific', kw))     
        return Pivot(self.conf, self.sparkConf, future, past)
    
    def contains(self, kw): 
        future = None
        past = None
        labels = ['emerging', 'decaying', 'stable', 'specific']
        attr = labels[0]
        if self.pivotFuture != None:
            conditionFuture = self.conditionArrayContains(self.pivotFuture, attr, kw)
            for a in labels[1:] :
                conditionFuture = conditionFuture | self.conditionArrayContains(self.pivotFuture, a, kw)
            future = self.pivotFuture.where(conditionFuture)
        if self.pivotPast != None:
            conditionPast = self.conditionArrayContains(self.pivotPast, attr, kw)
            for a in labels[1:] :
                conditionPast = conditionPast | self.conditionArrayContains(self.pivotPast, a, kw)
            past = self.pivotPast.where(conditionPast)
        return Pivot(self.conf, self.sparkConf, future, past)

    
    #### Temporal filters
    def period(self, year, restriction):
        future = None
        past = None
        if self.pivotFuture != None:
            if restriction == 0:
                future = self.pivotFuture.where(col('Startyear') >= year)
            else:
                future = self.pivotFuture.where(col('Endyear') <= year)
        if self.pivotPast != None:
            if restriction == 0:
                past = self.pivotPast.where(col('Startyear') >= year)  
            else:
                past = self.pivotPast.where(col('Endyear') <= year)  
        return Pivot(self.conf, self.sparkConf, future, past)
    
    
    #### Pattern filters
    def live(self, value, restriction):
        future = None
        past = None
        if self.pivotFuture != None:
            if restriction == 0:
                future = self.pivotFuture.where(col('Liveliness') >= value)
            else:
                future = self.pivotFuture.where(col('Liveliness') <= value)
        if self.pivotPast != None:
            if restriction == 0:
                past = self.pivotPast.where(col('Liveliness') >= value)  
            else:
                past = self.pivotPast.where(col('Liveliness') <= value)  
        return Pivot(self.conf, self.sparkConf, future, past)
    
    def split(self, value, restriction):
        future = None
        past = None
        if self.pivotFuture != None:
            if restriction == 0:
                future = self.pivotFuture.where(col('SplitDegree') >= value)
            else:
                future = self.pivotFuture.where(col('SplitDegree') <= value)
        if self.pivotPast != None:
            if restriction == 0:
                past = self.pivotPast.where(col('SplitDegree') >= value)  
            else:
                past = self.pivotPast.where(col('SplitDegree') <= value)  
        return Pivot(self.conf, self.sparkConf, future, past)
    
    def conv(self, value, restriction):
        future = None
        past = None
        if self.pivotFuture != None:
            if restriction == 0:
                future = self.pivotFuture.where(col('ConvergenceDegree') >= value)
            else:
                future = self.pivotFuture.where(col('ConvergenceDegree') <= value)
        if self.pivotPast != None:
            if restriction == 0:
                past = self.pivotPast.where(col('ConvergenceDegree') >= value)  
            else:
                past = self.pivotPast.where(col('ConvergenceDegree') <= value)  
        return Pivot(self.conf, self.sparkConf, future, past)
    
    
    #### Evolution Filters
    def Revol(self, value, restriction):
        future = None
        past = None
        if self.pivotFuture != None:
            if restriction == 0:
                future = self.pivotFuture.where(col('RelativeEvolutionDegree') >= value)
            else:
                future = self.pivotFuture.where(col('RelativeEvolutionDegree') <= value)
        if self.pivotPast != None:
            if restriction == 0:
                past = self.pivotPast.where(col('RelativeEvolutionDegree') >= value)  
            else:
                past = self.pivotPast.where(col('RelativeEvolutionDegree') <= value)  
        return Pivot(self.conf, self.sparkConf, future, past)
    
    def Pevol(self, value, restriction):
        future = None
        past = None
        if self.pivotFuture != None:
            if restriction == 0:
                future = self.pivotFuture.where(col('PivotEvolutionDegree') >= value)
            else:
                future = self.pivotFuture.where(col('PivotEvolutionDegree') <= value)
        if self.pivotPast != None:
            if restriction == 0:
                past = self.pivotPast.where(col('PivotEvolutionDegree') >= value)  
            else:
                past = self.pivotPast.where(col('PivotEvolutionDegree') <= value)  
        return Pivot(self.conf, self.sparkConf, future, past)
    
    
    #### Temporal projection
    def future(self):
        return Pivot(self.conf, self.sparkConf, self.pivotFuture, None)
    
    def past(self):
        return Pivot(self.conf, self.sparkConf, None, self.pivotPast)
    
    
    #### Set operations
    def union(self, anotherPivot):
        future = None
        past = None
        if self.pivotFuture != None:
            if anotherPivot.pivotFuture != None:
                future = self.pivotFuture.union(anotherPivot.pivotFuture)
            else:
                future = self.pivotFuture
        else:
            if anotherPivot.pivotFuture != None:
                future = anotherPivot.pivotFuture
                
        if self.pivotPast != None:
            if anotherPivot.pivotPast != None:
                past = self.pivotPast.union(anotherPivot.pivotPast)
            else:
                past = self.pivotPast
        else:
            if anotherPivot.pivotPast != None:
                past = anotherPivot.pivotPast
                
        return Pivot(self.conf, self.sparkConf, future, past)
    
    def intersect(self, anotherPivot):
        future = None
        past = None
        if self.pivotFuture != None and anotherPivot.pivotFuture != None:
            future = self.pivotFuture.intersect(anotherPivot.pivotFuture)
                
        if self.pivotPast != None and anotherPivot.pivotPast != None:
            past = self.pivotPast.intersect(anotherPivot.pivotPast)
                
        return Pivot(self.conf, self.sparkConf, future, past)
    
    def minus(self, anotherPivot):
        future = None
        past = None
        if self.pivotFuture != None:
            if anotherPivot.pivotFuture != None:
                future = self.pivotFuture.subtract(anotherPivot.pivotFuture)
            else:
                future = self.pivotFuture
                
        if self.pivotPast != None:
            if anotherPivot.pivotPast != None:
                past = self.pivotPast.subtract(anotherPivot.pivotPast)
            else:
                past = self.pivotPast
                
        return Pivot(self.conf, self.sparkConf, future, past)
    
    
    #### Path filters
    def path(self, anotherPivot):
        future = None
        past = None
        if self.pivotFuture != None:
            if anotherPivot.pivotFuture != None:
                topicsInPresentStats = self.pivotFuture.select(col("Beta"), col("TopicID").alias("Ti"))
                topicsInFutureStats = anotherPivot.pivotFuture.select(col("Beta"), col("TopicID").alias("Tk"))
                potentialPaths = topicsInPresentStats.join(topicsInFutureStats, "Beta").where("Ti < Tk")
                betas = [x[0] for x in potentialPaths.select("Beta").distinct().collect()]
                listOfPivots = [] 
                for beta in betas:
                    futurePivotGraphs = self.sparkConf.spark.read.json(self.evolutionPathAddr+str(beta)+'_'+str(self.conf.topicNumber)+'_0.0_10/future')
                    potentialPivotsByBeta = potentialPaths.where(col("Beta") == beta).select("Beta","Ti","Tk")
                    pivotsInPaths =potentialPivotsByBeta.intersect(futurePivotGraphs.select("Ti","Tk").withColumn("Beta", lit(beta)).select("Beta","Ti","Tk"))
                    pivotsInPathsDistinct = pivotsInPaths.select(col("Beta"), col("Ti").alias("TopicID")).distinct()
                    pivotsHavingPaths = self.pivotFuture.join(pivotsInPathsDistinct, ["Beta","TopicID"])
                    listOfPivots.append(pivotsHavingPaths)
                if len(listOfPivots) > 0:
                    future = reduce(lambda a,b : a.union(b), listOfPivots) 
        
        if self.pivotPast != None:
            if anotherPivot.pivotPast != None:
                topicsInPresentStats = self.pivotPast.select(col("Beta"), col("TopicID").alias("Ti"))
                topicsInPastStats = anotherPivot.pivotPast.select(col("Beta"), col("TopicID").alias("Tk"))
                potentialPaths = topicsInPresentStats.join(topicsInPastStats, "Beta").where("Tk < Ti")
                betas = [x[0] for x in potentialPaths.select("Beta").distinct().collect()]
                listOfPivots = [] 
                for beta in betas:
                    pastPivotGraphs = self.sparkConf.spark.read.json(self.evolutionPathAddr+str(beta)+'_'+str(self.conf.topicNumber)+'_0.0_10/past')
                    potentialPivotsByBeta = potentialPaths.where(col("Beta") == beta).select("Beta","Tk","Ti")
                    pivotsInPaths =potentialPivotsByBeta.intersect(pastPivotGraphs.select("Tk","Ti").withColumn("Beta", lit(beta)).select("Beta","Tk","Ti"))
                    pivotsInPathsDistinct = pivotsInPaths.select(col("Beta"), col("Ti").alias("TopicID")).distinct()
                    pivotsHavingPaths = self.pivotPast.join(pivotsInPathsDistinct, ["Beta","TopicID"])
                    listOfPivots.append(pivotsHavingPaths)
                if len(listOfPivots) > 0:
                    past = reduce(lambda a,b : a.union(b), listOfPivots) 
                
        return Pivot(self.conf, self.sparkConf, future, past)
    
    
    #### Ordering
    def sort(self, column, order = "asc"):
        future = None
        past = None
        if self.pivotFuture != None:
            if order == "asc":
                future = self.pivotFuture.orderBy(asc(column))
            else:
                future = self.pivotFuture.orderBy(desc(column))
        
        if self.pivotPast != None:
            if order == "asc":
                past = self.pivotPast.orderBy(asc(column))
            else:
                past = self.pivotPast.orderBy(desc(column))
        
        return Pivot(self.conf, self.sparkConf, future, past)
                
    
    def showLabels(self, numRows = 20, truncate = True):
        if self.pivotFuture != None:
            print('future pivot topics:')
            self.pivotFuture.select("Beta", "TopicID", "stable", "emerging", "decaying", "specific").distinct().show(numRows, truncate)
        if self.pivotPast != None:
            print('past pivot topics:')
            self.pivotPast.select("Beta", "TopicID", "stable", "emerging", "decaying", "specific").distinct().show(numRows, truncate)
            
    
    def describeTopics(self, numRows = 20, truncate = True):
        if self.pivotFuture != None:
            topics = self.sparkConf.spark.read.json(self.topicDictionaryFile+'_unique_10').withColumnRenamed("idTopic", "TopicID").withColumnRenamed("topic", "Terms")
            pivots = self.pivotFuture.select("TopicID").distinct()
            print('future pivot topics:')
            topics.join(pivots, "TopicID").show(numRows, truncate)
        
        if self.pivotPast != None:
            topics = self.sparkConf.spark.read.json(self.topicDictionaryFile+'_unique_10').withColumnRenamed("idTopic", "TopicID").withColumnRenamed("topic", "Terms")
            pivots = self.pivotPast.select("TopicID").distinct()
            print('past pivot topics:')
            topics.join(pivots, "TopicID").show(numRows, truncate)
            
            
    def describeTerms(self, numRows = 20, truncate = True):
        if self.pivotFuture != None:
            topics = self.sparkConf.spark.read.json(self.topicDictionaryFile+'_unique_10').withColumnRenamed("idTopic", "TopicID").withColumnRenamed("topic", "Terms")
            pivots = self.pivotFuture.select("TopicID").distinct()
            print('terms in future pivot topics:')
            topics.join(pivots, "TopicID").withColumn("Term", explode("Terms")).groupBy("Term").count().orderBy(desc("count")).show()
        
        if self.pivotPast != None:
            topics = self.sparkConf.spark.read.json(self.topicDictionaryFile+'_unique_10').withColumnRenamed("idTopic", "TopicID").withColumnRenamed("topic", "Terms")
            pivots = self.pivotPast.select("TopicID").distinct()
            print('terms in past pivot topics:')
            topics.join(pivots, "TopicID").withColumn("Term", explode("Terms")).groupBy("Term").count().orderBy(desc("count")).show()
       
    
    def describe(self, *columns):
        if self.pivotFuture != None:
            print('future pivot topics:')
            if len(columns) == 0:
                self.pivotFuture.select("Liveliness","SplitDegree","ConvergenceDegree","RelativeEvolutionDegree", "PivotEvolutionDegree").describe().show()
            else:
                self.pivotFuture.select(*columns).describe().show()
        if self.pivotPast != None:
            print('past pivot topics:')
            if len(columns) == 0:
                self.pivotPast.select("Liveliness","SplitDegree","ConvergenceDegree","RelativeEvolutionDegree", "PivotEvolutionDegree").describe().show()
            else:
                self.pivotPast.select(*columns).describe().show()
    
    
    def show(self, numRows = 20, truncate = True):
        if self.pivotFuture != None:
            print('future pivot topics:')
            self.pivotFuture.show(numRows, truncate)
        if self.pivotPast != None:
            print('past pivot topics:')
            self.pivotPast.show(numRows, truncate)