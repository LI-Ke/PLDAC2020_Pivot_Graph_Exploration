from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SQLContext
from pyspark import SparkContext as sc
from pyspark.sql.types import StructType
from pyspark.sql.functions import *
from functools import reduce

from graphviz import *
import numpy as np

from ipywidgets import interact, interactive, fixed, interact_manual, Button, HBox, VBox, Label
import ipywidgets as widgets

import time
import sys

class Pivot:
    def __init__(self, config, sparkConf, pivotFuture = None, pivotPast = None):
        self.conf = config
        self.sparkConf = sparkConf
        self.pivotFuture = pivotFuture
        self.pivotPast = pivotPast
        if self.conf.sourceCategory == 'all':
            self.phylomemyRoot = self.conf.dataPrefix + self.conf.storageDir + self.conf.algorithm + '/' + self.conf.source \
                + '/' + self.conf.corpusName + '/'+ str(self.conf.startYear) + str(self.conf.endYear) +str(self.conf.intervalYear) \
                + str(self.conf.stepYear) + '/' + str(self.conf.vocabSize) + '_' + str(self.conf.topicNumber) + '_' \
                + str(self.conf.maxIterations) + '_' + str(self.conf.maxTermNbrPerTopic) + '/'
        else:  
            self.phylomemyRoot = self.conf.dataPrefix + self.conf.storageDir + self.conf.algorithm + '/' + self.conf.source \
                + '/' + self.conf.corpusName + '/'+ self.conf.sourceCategory + '/' + str(self.conf.startYear) + str(self.conf.endYear) \
                + str(self.conf.intervalYear) + str(self.conf.stepYear) + '/' + str(self.conf.vocabSize) + '_' \
                + str(self.conf.topicNumber) + '_' + str(self.conf.maxIterations) + '_' + str(self.conf.maxTermNbrPerTopic) + '/'
        
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
                 
        self.periodList = self.sparkConf.spark.read.json(self.topicNodesFile).select("period").distinct().orderBy("period").collect()
        
        
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
                    futurePivotGraphs = self.sparkConf.spark.read.json(self.evolutionPathAddr+str(beta)+'_'+str(self.conf.topicNumber) \
                                                                       +'_0.0_10/future')
                    potentialPivotsByBeta = potentialPaths.where(col("Beta") == beta).select("Beta","Ti","Tk")
                    pivotsInPaths =potentialPivotsByBeta.intersect(futurePivotGraphs.select("Ti","Tk").withColumn("Beta", lit(beta)) \
                                                                   .select("Beta","Ti","Tk"))
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
                    pastPivotGraphs = self.sparkConf.spark.read.json(self.evolutionPathAddr+str(beta)+'_'+str(self.conf.topicNumber) \
                                                                     +'_0.0_10/past')
                    potentialPivotsByBeta = potentialPaths.where(col("Beta") == beta).select("Beta","Tk","Ti")
                    pivotsInPaths =potentialPivotsByBeta.intersect(pastPivotGraphs.select("Tk","Ti").withColumn("Beta", lit(beta)) \
                                                                   .select("Beta","Tk","Ti"))
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
            topics = self.sparkConf.spark.read.json(self.topicDictionaryFile+'_unique_10').withColumnRenamed("idTopic", "TopicID") \
            .withColumnRenamed("topic", "Terms")
            pivots = self.pivotFuture.select("TopicID").distinct()
            print('future pivot topics:')
            topics.join(pivots, "TopicID").show(numRows, truncate)
        
        if self.pivotPast != None:
            topics = self.sparkConf.spark.read.json(self.topicDictionaryFile+'_unique_10').withColumnRenamed("idTopic", "TopicID") \
            .withColumnRenamed("topic", "Terms")
            pivots = self.pivotPast.select("TopicID").distinct()
            print('past pivot topics:')
            topics.join(pivots, "TopicID").show(numRows, truncate)
            
            
    def describeTerms(self, numRows = 20, truncate = True):
        if self.pivotFuture != None:
            topics = self.sparkConf.spark.read.json(self.topicDictionaryFile+'_unique_10').withColumnRenamed("idTopic", "TopicID") \
            .withColumnRenamed("topic", "Terms")
            pivots = self.pivotFuture.select("TopicID").distinct()
            print('terms in future pivot topics:')
            topics.join(pivots, "TopicID").withColumn("Term", explode("Terms")).groupBy("Term").count().orderBy(desc("count")).show()
        
        if self.pivotPast != None:
            topics = self.sparkConf.spark.read.json(self.topicDictionaryFile+'_unique_10').withColumnRenamed("idTopic", "TopicID") \
            .withColumnRenamed("topic", "Terms")
            pivots = self.pivotPast.select("TopicID").distinct()
            print('terms in past pivot topics:')
            topics.join(pivots, "TopicID").withColumn("Term", explode("Terms")).groupBy("Term").count().orderBy(desc("count")).show()
       
    
    def describe(self, *columns):
        if self.pivotFuture != None:
            print('future pivot topics:')
            if len(columns) == 0:
                self.pivotFuture.select("Liveliness","SplitDegree","ConvergenceDegree","RelativeEvolutionDegree", "PivotEvolutionDegree") \
                .describe().show()
            else:
                self.pivotFuture.select(*columns).describe().show()
        if self.pivotPast != None:
            print('past pivot topics:')
            if len(columns) == 0:
                self.pivotPast.select("Liveliness","SplitDegree","ConvergenceDegree","RelativeEvolutionDegree", "PivotEvolutionDegree") \
                .describe().show()
            else:
                self.pivotPast.select(*columns).describe().show()
    

    def show(self, numRows = 20, truncate = True):
        if self.pivotFuture != None:
            print('future pivot topics:')
            self.pivotFuture.show(numRows, truncate)
        if self.pivotPast != None:
            print('past pivot topics:')
            self.pivotPast.show(numRows, truncate)
    
    #### graphic panel
    #### query processing panel
    def query_form(self):
        def my_max(list):
            list.sort()
            return (list[0])
        
        maxLive = 0
        maxLiveFuture = 0
        maxLivePast = 0
        maxOut = 1.0
        maxOutFuture = 1.0
        maxOutPast = 1.0
        maxIn = 1.0
        maxInFuture = 1.0
        maxInPast = 1.0
        
        keywordsBFValue = ['---Select a keyword---']
        stableBFValue = ['---Select a stable term---']
        emergingBFValue = ['---Select an emerging term---']
        decayingBFValue = ['---Select a decaying term---']
        specificBFValue = ['---Select a specific term---']
        
        stableOptions = []
        emergingOptions = []
        decayingOptions = []
        specificOptions = []
        
        if self.pivotFuture != None:
            maxLiveFuture = self.pivotFuture.select("Startyear").distinct().count() - 1
            maxMetricsFuture = self.pivotFuture.describe("SplitDegree", "ConvergenceDegree").filter("summary = 'max'").select("SplitDegree", "ConvergenceDegree").first().asDict()
            maxOutFuture = maxMetricsFuture['SplitDegree']
            maxInFuture = maxMetricsFuture['ConvergenceDegree']
            futurelabels = self.pivotFuture.select("stable","emerging","decaying","specific").collect()
            stableOptions = stableOptions + [term for row in futurelabels for term in row.stable]
            emergingOptions = emergingOptions + [term for row in futurelabels for term in row.emerging]
            decayingOptions = decayingOptions + [term for row in futurelabels for term in row.decaying]
            specificOptions = specificOptions + [term for row in futurelabels for term in row.specific]
        if self.pivotPast != None:
            maxLivePast = self.pivotPast.select("Startyear").distinct().count() - 1
            maxMetricsPast = self.pivotPast.describe("SplitDegree", "ConvergenceDegree").filter("summary = 'max'").select("SplitDegree", "ConvergenceDegree").first().asDict()
            maxOutPast = maxMetricsPast['SplitDegree']
            maxInPast = maxMetricsPast['ConvergenceDegree']
            pastlabels = self.pivotPast.select("stable","emerging","decaying","specific").collect()
            stableOptions = stableOptions + [term for row in pastlabels for term in row.stable]
            emergingOptions = emergingOptions + [term for row in pastlabels for term in row.emerging]
            decayingOptions = decayingOptions + [term for row in pastlabels for term in row.decaying]
            specificOptions = specificOptions + [term for row in pastlabels for term in row.specific]
                
        maxLive = my_max([maxLiveFuture, maxLivePast])
        maxOut = my_max([maxOutFuture, maxOutPast])
        maxIn = my_max([maxInFuture, maxInPast])
        
        keywordsOptions = list(set(stableOptions + emergingOptions + decayingOptions + specificOptions))
        keywordsOptions.sort()
        keywordsOptions = keywordsBFValue + keywordsOptions
        
        stableOptions = list(set(stableOptions))
        stableOptions.sort()
        stableOptions = stableBFValue + stableOptions     
        
        emergingOptions = list(set(emergingOptions))
        emergingOptions.sort()
        emergingOptions = emergingBFValue + emergingOptions
        
        decayingOptions = list(set(decayingOptions))
        decayingOptions.sort()
        decayingOptions = decayingBFValue + decayingOptions
        
        specificOptions = list(set(specificOptions))
        specificOptions.sort()
        specificOptions = specificBFValue + specificOptions
                                        
        startyear = self.conf.startYear
        endyear = self.conf.endYear

        style = {'description_width': 'initial'}

#         keywordWidget = widgets.Text(placeholder='Type keywords:', description='Keywords')
#         stableWidget = widgets.Text(placeholder='Search by stable keywords', description='stable')
#         emergingWidget = widgets.Text(placeholder='Search by emerging keywords', description='emerging')
#         decayingWidget = widgets.Text(placeholder='Search by decaying keywords', description='decaying')
#         specificWidget = widgets.Text(placeholder='Search by specific keywords', description='specific')

        keywordWidget = widgets.Dropdown(
            options=keywordsOptions,
            value='---Select a keyword---',
            description='Keywords:',
            disabled=False,
        )
    
        stableWidget = widgets.Dropdown(
            options=stableOptions,
            value='---Select a stable term---',
            description='stable',
            disabled=False,
        )
        
        emergingWidget = widgets.Dropdown(
            options=emergingOptions,
            value='---Select an emerging term---',
            description='emerging',
            disabled=False,
        )
        
        decayingWidget = widgets.Dropdown(
            options=decayingOptions,
            value='---Select a decaying term---',
            description='decaying',
            disabled=False,
        )
        
        specificWidget = widgets.Dropdown(
            options=specificOptions,
            value='---Select a specific term---',
            description='specific',
            disabled=False,
        )

        periodWidget = widgets.IntRangeSlider(
            value=[startyear, endyear],
            min=startyear,
            max=endyear,
            step=1,
            description='Period:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            style=style)
        periodWidget.style.handle_color = 'lightblue'
        
        projecFutureWidget =widgets.Checkbox(
            value=False,
            description='future',
            disabled=False
        )

        projecPastWidget =widgets.Checkbox(
            value=False,
            description='past',
            disabled=False
        )

        livelinessWidget = widgets.IntRangeSlider(
            value=[0, maxLive],
            min=0,
            max=maxLive,
            step=1,
            description='Liveliness:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            style=style)
        livelinessWidget.style.handle_color = 'lightblue'

        outDegreeWidget = widgets.FloatRangeSlider(
            value=[1, maxOut],
            min=1,
            max=maxOut,
            step=0.1,
            description='Split degree:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            style=style)
        outDegreeWidget.style.handle_color = 'lightblue'

        inDegreeWidget = widgets.FloatRangeSlider(
            value=[1, maxIn],
            min=1,
            max=maxIn,
            step=0.1,
            description='Convergence degree:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            style=style)
        inDegreeWidget.style.handle_color = 'lightblue'

        relEvolutionWidget = widgets.FloatRangeSlider(
            value=[0, 1.0],
            min=0,
            max=1.0,
            step=0.1,
            description='Relative evolution degree:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            style=style)
        relEvolutionWidget.style.handle_color = 'lightblue'

        pivotEvolutionWidget = widgets.FloatRangeSlider(
            value=[0, 1.0],
            min=0,
            max=1.0,
            step=0.1,
            description='Pivot evolution degree:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            style=style)
        pivotEvolutionWidget.style.handle_color = 'lightblue'

        def setQueryParam(text, emerging, decaying, stable, specific, period, projecFuture, projecPast, liveliness, \
                          outDegree, inDegree, relEvolution, pivotEvolution):
            return
        
        output = widgets.interactive_output(setQueryParam, {'text': keywordWidget, 'emerging': emergingWidget, 'decaying': decayingWidget, \
                                                            'stable': stableWidget, 'specific': specificWidget, 'period': periodWidget, \
                                                            'projecFuture': projecFutureWidget, 'projecPast': projecPastWidget, \
                                                            'liveliness': livelinessWidget, 'outDegree': outDegreeWidget, 'inDegree': inDegreeWidget, \
                                                            'relEvolution': relEvolutionWidget, 'pivotEvolution': pivotEvolutionWidget})

        hbox1 = HBox([stableWidget, emergingWidget])
        hbox2 = HBox([decayingWidget, specificWidget])
        hbox3 = HBox([Label('Projection'), projecFutureWidget, projecPastWidget])
        hbox4 = HBox([outDegreeWidget, inDegreeWidget])
        hbox6 = HBox([relEvolutionWidget, pivotEvolutionWidget])
        ui = VBox([keywordWidget, hbox1, hbox2, periodWidget, hbox3, livelinessWidget, hbox4, hbox6])

        display(ui, output)


        def processQuery():
            currentPivots = Pivot(self.conf, self.sparkConf, self.pivotFuture, self.pivotPast)
            if keywordWidget.value != "---Select a keyword---":
#                 print('contains: ' + keywordWidget.value)
                currentPivots = currentPivots.contains(keywordWidget.value)
            if stableWidget.value != "---Select a stable term---":
#                 print('stable: ' + stableWidget.value )
                currentPivots = currentPivots.stable(stableWidget.value)
            if emergingWidget.value != "---Select an emerging term---":
#                 print('emerging: ' + emergingWidget.value)
                currentPivots = currentPivots.emerge(emergingWidget.value)
            if decayingWidget.value != "---Select a decaying term---":
#                 print('decaying: ' + decayingWidget.value)
                currentPivots = currentPivots.decay(decayingWidget.value)
            if specificWidget.value != "---Select a specific term---":
#                 print('specific: ' + specificWidget.value)     
                currentPivots = currentPivots.specific(specificWidget.value)
            currentPivots = currentPivots.period(periodWidget.value[0],0).period(periodWidget.value[1],1)
            if (projecFutureWidget.value == True and projecPastWidget.value == False):
                currentPivots = currentPivots.future().live(livelinessWidget.value[0], 0).live(livelinessWidget.value[1],1) \
                    .split(outDegreeWidget.value[0],0).split(outDegreeWidget.value[1],1) \
                    .conv(inDegreeWidget.value[0],0).conv(inDegreeWidget.value[1],1) \
                    .Revol(relEvolutionWidget.value[0],0).Revol(relEvolutionWidget.value[1],1) \
                    .Pevol(pivotEvolutionWidget.value[0],0).Pevol(pivotEvolutionWidget.value[1],1)
            elif (projecFutureWidget.value == False and projecPastWidget.value == True):
                currentPivots = currentPivots.past().live(livelinessWidget.value[0], 0).live(livelinessWidget.value[1],1) \
                    .split(outDegreeWidget.value[0],0).split(outDegreeWidget.value[1],1) \
                    .conv(inDegreeWidget.value[0],0).conv(inDegreeWidget.value[1],1) \
                    .Revol(relEvolutionWidget.value[0],0).Revol(relEvolutionWidget.value[1],1) \
                    .Pevol(pivotEvolutionWidget.value[0],0).Pevol(pivotEvolutionWidget.value[1],1)
            else:
                currentPivots = currentPivots.live(livelinessWidget.value[0], 0).live(livelinessWidget.value[1],1) \
                    .split(outDegreeWidget.value[0],0).split(outDegreeWidget.value[1],1) \
                    .conv(inDegreeWidget.value[0],0).conv(inDegreeWidget.value[1],1) \
                    .Revol(relEvolutionWidget.value[0],0).Revol(relEvolutionWidget.value[1],1) \
                    .Pevol(pivotEvolutionWidget.value[0],0).Pevol(pivotEvolutionWidget.value[1],1)
            self.pivotFuture = currentPivots.pivotFuture
            self.pivotPast = currentPivots.pivotPast
        
        
        def on_button_clicked(b):
            with out:
                out.clear_output()
                processQuery()
                print("Query processing done!")
        
        button = widgets.Button(description="Search")
        button.style.button_color = 'lightblue'
        
        button.on_click(on_button_clicked)
        out = widgets.Output()
        display(button, out)
    
    #### graph visualization panel
    # define timeline graph funtion
    def generateSVGSubgraphTimeline(self, graph, subgraph):
        pj = subgraph.select(col("Pj").alias("p"))
        pk = subgraph.select(col("Pk").alias("p"))
        p = pj.union(pk).distinct()
    
        # draw timeline
        # period nodes
        for periodId in p.collect():
            graph.node('P'+str(periodId.p), shape="plaintext", label=self.periodList[periodId[0]][0])
    
        # period edges
        edges = subgraph.select('Pj', 'Pk').distinct()
        for e in edges.collect():
            if (e.Pj < e.Pk):
                graph.edge('P'+str(e.Pj), 'P'+str(e.Pk))
            else:
                graph.edge('P'+str(e.Pk), 'P'+str(e.Pj))
            
        return graph

    # g = Digraph('g2', format='png')
    # graphTimeline = generateSVGSubgraphTimeline(g, subgraphFuture, periodList)
    # graphTimeline

    # label topics
    def topicLabeling(self, graph, labels):
        for row in labels.select("Ti","stable", "emerging", "decaying", "specific").collect():
            label = "<table border=\"0\" cellspacing=\"0\">" + \
            "\n                    <tr><td border=\"1\">id:"+ str(row.Ti) + "</td></tr>" + \
            "\n                    <tr><td border=\"1\" bgcolor=\"deepskyblue\">"+ "<br/>".join(row.stable) + "</td></tr>" + \
            "\n                    <tr><td border=\"1\" bgcolor=\"green2\">"+ "<br/>".join(row.emerging) + "</td></tr>" + \
            "\n                    <tr><td border=\"1\" bgcolor=\"red2\">"+ "<br/>".join(row.decaying) + "</td></tr>" + \
            "\n                    <tr><td border=\"1\">"+"<br/>".join(row.specific) + "</td></tr>" +"\n                </table>"
            graph.node(str(row.Ti), shape="none", label="<" + label + ">")
    
        return graph

    # align subnodes in pivot graph
    def subgraphLabeling(self, graph, beta, timeDirection, subgraph, pivotLabels):
        # draw timeline
        graphTimeline = self.generateSVGSubgraphTimeline(graph, subgraph)
    
        parentsInPivotGraph = [ (x.Tj) for x in subgraph.select("Tj").distinct().collect()]
        childrenInPivotGraph = [ (x.Tk) for x in subgraph.select("Tk").distinct().collect()]
        topicsInPivotGraph = parentsInPivotGraph + childrenInPivotGraph
        subNodesLabels = pivotLabels.filter(col("Ti").isin(topicsInPivotGraph))
        graphTopicLabels = self.topicLabeling(graphTimeline, subNodesLabels)
    
        if (timeDirection == "future"):
            for e in subgraph.select( "Tj", "Tk", "Beta").distinct().collect():
                width = str((np.round(e.Beta, 2)-beta)*(10-1)/(1-beta)+1)
                graphTopicLabels.edge(str(e.Tj), str(e.Tk), penwidth=width, label=str(np.round(e.Beta, 2)), fontcolor='red')
        else:
            for e in subgraph.select( "Tk", "Tj", "Beta").distinct().collect():
                width = str((np.round(e.Beta, 2)-beta)*(10-1)/(1-beta)+1)
                graphTopicLabels.edge(str(e.Tk), str(e.Tj), penwidth=width, label=str(np.round(e.Beta, 2)), fontcolor='red')
    
        # align topic nodes {rank: same}
    
        return graphTopicLabels

    # label individual pivot topic
    def individualTopicLabeling(self, graph, individuals):
        if (individuals.count() != 0):
            for row in individuals.select("id", "topic", "period").collect():
                lab = "<table border=\"0\" cellspacing=\"0\">" + \
                "\n                    <tr><td border=\"1\">id:"+ str(row.id) + "</td></tr>" + \
                "\n                    <tr><td border=\"1\">"+ "<br/>".join(row.topic) + "</td></tr>" + \
                "\n                </table>"
                graph.node('P'+str(row.period), shape="plaintext", label=self.periodList[row.period][0])
                graph.node(str(row.id), shape="none", label="<" + lab + ">")
        return graph

    # draw bidirectional pivot graph   
    def bidirectionPivotGraph(self, graph, beta, subgraphFuture, subgraphPast, pivotLabels):
        # draw timeline
        graph1 = self.generateSVGSubgraphTimeline(graph, subgraphFuture)
        graph2 = self.generateSVGSubgraphTimeline(graph1, subgraphPast)
    
        parentsInPivotGraphFuture = [ (x.Tj) for x in subgraphFuture.select("Tj").distinct().collect()]
        childrenInPivotGraphFuture = [ (x.Tk) for x in subgraphFuture.select("Tk").distinct().collect()]
        parentsInPivotGraphPast = [ (x.Tj) for x in subgraphPast.select("Tj").distinct().collect()]
        childrenInPivotGraphPast = [ (x.Tk) for x in subgraphPast.select("Tk").distinct().collect()]
        topicsInPivotGraph = parentsInPivotGraphFuture + childrenInPivotGraphFuture + parentsInPivotGraphPast + childrenInPivotGraphPast
        # label topics in pivot graph
        subNodesLabels = pivotLabels.filter(col("Ti").isin(topicsInPivotGraph))
        graphTopicLabels = self.topicLabeling(graph2, subNodesLabels)
    
        nodeLinksFuture = subgraphFuture.select( col("Tj").alias("From"), col("Tk").alias("To"), col("Beta")).distinct()
        nodeLinksPast = subgraphPast.select( col("Tk").alias("From"), col("Tj").alias("To"), col("Beta")).distinct()
        nodeLinks = nodeLinksFuture.union(nodeLinksPast)
        for e in nodeLinks.collect():
            width = str((np.round(e.Beta, 2)-beta)*(10-1)/(1-beta)+1)
            graphTopicLabels.edge(str(e.From), str(e.To), penwidth=width, label=str(np.round(e.Beta, 2)), fontcolor='red')
    
        # align topic nodes {rank: same}   
    
        return graphTopicLabels
    
    
    def drawPivotGraph(self, topicId, beta, pivotGraphs):
        subgraphFuture = pivotGraphs[beta][0].where(col("Ti") == topicId)
        subgraphPast = pivotGraphs[beta][1].where(col("Ti") == topicId)
        pivotLabels =  pivotGraphs[beta][2]
    
        # g = Digraph('g2', format='svg')
        g = Digraph('G', format='png', strict=True)
        if ((subgraphFuture.count() == 0) and (subgraphPast.count() > 0)):
            g = self.subgraphLabeling(g, beta, "past", subgraphPast, pivotLabels)
        elif ((subgraphFuture.count() > 0) and (subgraphPast.count() == 0)):
            g = self.subgraphLabeling(g, beta, "future", subgraphFuture, pivotLabels)
        elif ((subgraphFuture.count() == 0) and (subgraphPast.count() == 0)):
            individuals = self.sparkConf.spark.read.json(fileRoot+str(beta)+'_30_0.0_10/Global_individuals_future')
            individualTopic = individuals.where(col("id") == topicId)
            g = self.individualTopicLabeling(g, individualTopic)
        else:
            g = self.bidirectionPivotGraph(g, beta, subgraphFuture, subgraphPast, pivotLabels)

        return g
    
    def drawPivotGraphs(self, pivots):
        pivotToDraw = None
        topics = None
        pivotGraphs = {}
        if pivots.pivotFuture != None and pivots.pivotPast != None:
            topics = pivots.pivotFuture.select("TopicID", "Beta").union(pivots.pivotPast.select("TopicID", "Beta")).distinct()
        elif pivots.pivotFuture != None and pivots.pivotPast == None:
            topics = pivots.pivotFuture.select("TopicID", "Beta")
        elif pivots.pivotFuture == None and pivots.pivotPast != None:
            topics = pivots.pivotPast.select("TopicID", "Beta")
        
        if topics != None:
            betas = [x[0] for x in topics.select("Beta").distinct().collect()]
            for beta in betas:
                subgraphFuture = self.sparkConf.spark.read.json(self.evolutionPathAddr+str(beta)+'_20_0.0_10/future').persist()
                subgraphPast = self.sparkConf.spark.read.json(self.evolutionPathAddr+str(beta)+'_20_0.0_10/past').persist()
                pivotLabels =  self.sparkConf.spark.read.json(self.evolutionPathAddr+str(beta)+'_20_0.0_10/labels').persist()
                pivotGraphs[beta] = (subgraphFuture, subgraphPast, pivotLabels)
            pivotToDraw = [ (x.TopicID, x.Beta) for x in topics.collect()]
        
        if len(pivotToDraw) > 0:
            return {str(x[0])+'('+str(x[1])+')': self.drawPivotGraph(x[0], x[1], pivotGraphs) for x in pivotToDraw}
        else:
            return None
        
    def drawMultiPivotGraph(self, beta, pivotGraphs, betaTopics):
        subgraphFuture = pivotGraphs[beta][0].where(col("Ti").isin(betaTopics))
        subgraphPast = pivotGraphs[beta][1].where(col("Ti").isin(betaTopics))
        pivotLabels =  pivotGraphs[beta][2]
    
        # g = Digraph('g2', format='svg')
        g = Digraph('G2', format='png', strict=True)
        if ((subgraphFuture.count() == 0) and (subgraphPast.count() > 0)):
            g = self.subgraphLabeling(g, beta, "past", subgraphPast, pivotLabels)
        elif ((subgraphFuture.count() > 0) and (subgraphPast.count() == 0)):
            g = self.subgraphLabeling(g, beta, "future", subgraphFuture, pivotLabels)
        elif ((subgraphFuture.count() == 0) and (subgraphPast.count() == 0)):
            individuals = self.sparkConf.spark.read.json(fileRoot+str(beta)+'_30_0.0_10/Global_individuals_future')
            individualTopic = individuals.where(col("id").isin(betaTopics))
            g = self.individualTopicLabeling(g, individualTopic)
        else:
            g = self.bidirectionPivotGraph(g, beta, subgraphFuture, subgraphPast, pivotLabels)

        return g
    
    def drawMultiPivotGraphs(self, pivots):
        topics = None
        pivotGraphs = {}
        betaTopics={}
        if pivots.pivotFuture != None and pivots.pivotPast != None:
            topics = pivots.pivotFuture.select("TopicID", "Beta").union(pivots.pivotPast.select("TopicID", "Beta")).distinct()
        elif pivots.pivotFuture != None and pivots.pivotPast == None:
            topics = pivots.pivotFuture.select("TopicID", "Beta")
        elif pivots.pivotFuture == None and pivots.pivotPast != None:
            topics = pivots.pivotPast.select("TopicID", "Beta")
        
        if topics != None:
            betaTopics = {row.Beta: row.TopicID for row in topics.groupBy("Beta").agg(collect_list("TopicID").alias("TopicID")).collect()}
            betas = [x[0] for x in topics.select("Beta").distinct().collect()]
            for beta in betas:
                subgraphFuture = self.sparkConf.spark.read.json(self.evolutionPathAddr+str(beta)+'_20_0.0_10/future').persist()
                subgraphPast = self.sparkConf.spark.read.json(self.evolutionPathAddr+str(beta)+'_20_0.0_10/past').persist()
                pivotLabels =  self.sparkConf.spark.read.json(self.evolutionPathAddr+str(beta)+'_20_0.0_10/labels').persist()
                pivotGraphs[beta] = (subgraphFuture, subgraphPast, pivotLabels)
        
        if len(pivotGraphs) > 0:
            return {str(x): self.drawMultiPivotGraph(x, pivotGraphs, betaTopics[x]) for x in betas}
        else:
            return None
    
    def visualize(self):
        pivots = Pivot(self.conf, self.sparkConf, self.pivotFuture, self.pivotPast)
        graphs = self.drawPivotGraphs(pivots)
        multigraphs = self.drawMultiPivotGraphs(pivots)
        def on_button_clicked1(b):
            with out:
                out.clear_output()
                keys = list(graphs.keys())
                keys.sort()
                def getGraph(pivot, size):
                    g = graphs[pivot]
                    g.attr(size = str(size))
                    return g

                topicWidget = widgets.ToggleButtons(
                    options=keys,
                    description='Pivot topics:',
                    disabled=False,
                    button_style='info')

                sizeWidget = widgets.IntSlider(
                    value=4,
                    min=1,
                    max=40,
                    step=1,
                    description='Image size:',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='d')
                interact(getGraph, pivot = topicWidget, size = sizeWidget)
        
        button1 = widgets.Button(description="Single pivot graphs")
        button1.style.button_color = 'lightblue'
        
        button1.on_click(on_button_clicked1)
        
        def on_button_clicked2(b):
            with out:
                out.clear_output()
                keys = list(multigraphs.keys())
                keys.sort()
                def getGraph(pivot, size):
                    g = multigraphs[pivot]
                    g.attr(size = str(size))
                    return g

                topicWidget = widgets.ToggleButtons(
                    options=keys,
                    description='Betas:',
                    disabled=False,
                    button_style='info')

                sizeWidget = widgets.IntSlider(
                    value=4,
                    min=1,
                    max=40,
                    step=1,
                    description='Image size:',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='d')
                interact(getGraph, pivot = topicWidget, size = sizeWidget)
                
        button2 = widgets.Button(description="Multiple pivot graphs")
        button2.style.button_color = 'lightblue'
        
        button2.on_click(on_button_clicked2)
        out = widgets.Output()
#         display(button1, out)
        ui = HBox([button1, button2])
        display(ui, out)