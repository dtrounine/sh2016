/**
  * Created by dmitry.trunin on 07.04.2016.
  */
object Paths {
    def getGraphPath(dataDir: String) = dataDir + "trainGraph"
    def getGraphInteractionPath(dataDir: String) = dataDir + "interactionGraph"
    def getReversedGraphPath(dataDir: String) = dataDir + "trainSubReversedGraph"
    def getReversedGraphTxtPath(dataDir: String) = dataDir + "trainSubReversedGraph_txt"

    def getCommonFriendsPath(dataDir: String) = dataDir + "commonFriendsPartitioned"
    def getCommonFriendsPartPath(dataDir: String, partNo: Int) = getCommonFriendsPath(dataDir) + "/part_" + partNo
    def getAllCommonFriendsPartsPathMask(dataDir: String) = getCommonFriendsPath(dataDir) + "/part_*"
    def getCommonFriendsTextPath(dataDir: String) = dataDir + "commonFriendsPartitioned_txt"
    def getCommonFriendsPartTextPath(dataDir: String, partNo: Int) = getCommonFriendsTextPath(dataDir) + "/part_" + partNo

    def getDemographyPath(dataDir: String) = dataDir + "demography"
    def getPredictionPath(dataDir: String) = dataDir + "prediction"
    def getTrainDataPath(dataDir: String) = dataDir + "trainData"
    def getModelPath(dataDir: String) = dataDir + "LogisticRegressionModel"
    def getCityPairCountPath(dataDir: String) = dataDir + "cityPairCount"
    def getCityPairCountTxtPath(dataDir: String) = getCityPairCountPath(dataDir) + "_txt"
    def getCityPopulationPath(dataDir: String) = dataDir + "cityPopulation"
    def getCityPopulationTxtPath(dataDir: String) = getCityPopulationPath(dataDir) + "_txt"
    def getUserPageRankPath(dataDir: String) = dataDir + "userPageRank"
    def getUserPageRankTxtPath(dataDir: String) = getUserPageRankPath(dataDir) + "_txt"
    def getOtherDetailsPath(dataDir: String) = dataDir + "otherDetails"
    def getInteractionsPath(dataDir: String) = dataDir + "interactions"

    def getFeaturesPath(dataDir: String) = dataDir + "features"
}
