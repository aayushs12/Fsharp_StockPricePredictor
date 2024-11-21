module APIVersion

open System
open Deedle
open MathNet.Numerics.LinearAlgebra
open Accord.Statistics.Models.Regression.Linear
open Suave
open Suave.Filters
open Suave.Operators
open Suave.Successful
open Suave.RequestErrors
open Suave.ServerErrors
open Suave.Files
open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Transforms.TimeSeries
open XPlot.GoogleCharts

let staticFilesPath = Path.Combine(__SOURCE_DIRECTORY__, "StaticFiles")
Directory.CreateDirectory(staticFilesPath) |> ignore


type PriceData () =
    [<DefaultValue>]
    [<LoadColumn(0)>]
    val mutable public Date:string
    [<DefaultValue>]
    [<LoadColumn(1)>]
    val mutable public Open:float32
    [<DefaultValue>]
    [<LoadColumn(2)>]
    val mutable public High:float32
    [<DefaultValue>]
    [<LoadColumn (3)>]
    val mutable public Low:float32
    [<DefaultValue>]
    [<LoadColumn(4)>]
    val mutable public Close:float32
    [<DefaultValue>]
    [<LoadColumn (5)>]
    val mutable public AdjClose:float32
    [<DefaultValue>]
    [<LoadColumn (6)>]
    val mutable public Volume:float32

type PricePrediction () =
    [<DefaultValue>]
    val mutable public Date:string
    [<DefaultValue>]
    val mutable public Prediction:double[]

// Placeholder for results to be shared across API calls
let mutable predictionsResult: (float * float * (float[] * float[])) option = None
let mutable nextPriceResult: (float * string) option = None

// Load and preprocess data
let loadData (filePath: string) =
    let df = Frame.ReadCsv(filePath)
    let requiredColumns = ["Date"; "Open"; "High"; "Low"; "Close"; "Volume"; "Adj Close"]
    if not (requiredColumns |> List.forall (fun col -> df.ColumnKeys |> Seq.contains col)) then
        failwith "Required columns not found in the dataset."
    df
let calculateMACD (series: Series<int, float>) (shortWindow: int) (longWindow: int) (signalWindow: int) =
    let shortEMA = Stats.movingMean shortWindow series
    let longEMA = Stats.movingMean longWindow series
    let macd = shortEMA - longEMA
    let signal = Stats.movingMean signalWindow macd
    macd - signal

let calculateRSI (series: Series<int, float>) (period: int) =
    let diff = series - Series.shift 1 series
    let gain = diff |> Series.mapValues (fun x -> if x > 0.0 then x else 0.0)
    let loss = diff |> Series.mapValues (fun x -> if x < 0.0 then -x else 0.0)
    let avgGain = Stats.movingMean period gain
    let avgLoss = Stats.movingMean period loss
    avgGain / (avgGain + avgLoss) * 100.0

let createFeaturesAndLabels (df: Frame<int, string>) =
    let labels = df.GetColumn<float>("Adj Close") |> Series.values |> Array.ofSeq
    let laggedFeatures =
        [1 .. 5]
        |> List.map (fun lag -> df.GetColumn<float>("High") |> Series.shift -lag |> Series.values |> Array.ofSeq)
        |> Array.ofList 
    // Calculate MACD and RSI indicators and add them as features
    let closeSeries = df.GetColumn<float>("Close")
    let macdSeries = calculateMACD closeSeries 12 26 9 |> Series.values |> Array.ofSeq
    let rsiSeries = calculateRSI closeSeries 14 |> Series.values |> Array.ofSeq

    // Combine all features into a single array
    let allFeatures =
        Array.concat [| laggedFeatures; [| macdSeries; rsiSeries |] |]

    let minLength = allFeatures |> Array.minBy Array.length |> Array.length
    let trimmedFeatures = allFeatures |> Array.map (fun feature -> feature.[0..minLength - 1])
    let trimmedLabels = labels.[5..minLength + 4]
    let features = Array.transpose trimmedFeatures
    let validData =
        Array.zip features trimmedLabels
        |> Array.filter (fun (f, l) -> not (Array.exists Double.IsNaN f || Double.IsNaN l))
    Array.unzip validData

let splitData (features: float[][]) (labels: float[]) =
    let n = features.Length
    let trainSize = int (0.8 * float n)
    let trainX, testX = Array.splitAt trainSize features
    let trainY, testY = Array.splitAt trainSize labels
    (trainX, trainY, testX, testY)

let trainLinearRegression (trainX: float[][]) (trainY: float[]) =
    let model = MultipleLinearRegression(trainX.[0].Length)
    model.Regress(trainX, trainY)
    model

let meanSquaredError (actual: float[]) (predicted: float[]) =
    actual
    |> Array.zip predicted
    |> Array.map (fun (y, yPred) -> (y - yPred) ** 2.0)
    |> Array.average

let coefficientOfDetermination (actual: float[]) (predicted: float[]) =
    let mean = actual |> Array.average
    let totalSumOfSquares = actual |> Array.sumBy (fun y -> (y - mean) ** 2.0)
    let residualSumOfSquares = actual |> Array.zip predicted |> Array.sumBy (fun (y, yPred) -> (y - yPred) ** 2.0)
    1.0 - (residualSumOfSquares / totalSumOfSquares)

let evaluateLinearRegression (model: MultipleLinearRegression) (testX: float[][]) (testY: float[]) =
    let predictions = testX |> Array.map model.Transform
    let mse = meanSquaredError testY predictions
    let r2 = coefficientOfDetermination testY predictions
    (mse, r2, (testY, predictions))

let runPredictionPipeline (filePath: string) =
    let data = loadData filePath
    let features, labels = createFeaturesAndLabels data
    let trainX, trainY, testX, testY = splitData features labels

    let lrModel = trainLinearRegression trainX trainY
    let mse, r2, results = evaluateLinearRegression lrModel testX testY
    predictionsResult <- Some (mse, r2, results)

    let lastFeatures = features |> Array.last
    let nextPrice = lrModel.Transform(lastFeatures)
    let lastDate = data.GetColumn<DateTime>("Date").Values |> Seq.last
    let nextDate = lastDate.AddDays(1.0).ToString("yyyy-MM-dd")
    nextPriceResult <- Some (nextPrice, nextDate)

let generateGraph (filePath: string) =
    let ctx = MLContext()
    let dataView = ctx.Data.LoadFromTextFile<PriceData>(path = filePath, hasHeader = true, separatorChar = ',')
    let anomalyPValueHistoryLength = 30
    let changePointPValueHistoryLength = 10
    let anomalyConfidence = 95
    let changePointConfidence = 95
    let anomalyPipeline =
        ctx
            .Transforms
            .DetectIidSpike(
                outputColumnName = "Prediction",
                inputColumnName = "Close",
                side = AnomalySide.TwoSided,
                confidence = anomalyConfidence,
                pvalueHistoryLength = anomalyPValueHistoryLength)

    let changePointPipeLine =
        ctx
            .Transforms
            .DetectIidChangePoint(
                outputColumnName = "Prediction",
                inputColumnName = "Close",
                martingale = MartingaleType.Power,
                confidence = changePointConfidence,
                changeHistoryLength = changePointPValueHistoryLength)


    let trainedAnomalyModel = anomalyPipeline.Fit(dataView)
    let trainedChangePointModel = changePointPipeLine.Fit(dataView)
    let transformedAnomalyData = trainedAnomalyModel.Transform(dataView)
    let transformedChangePointData = trainedChangePointModel.Transform(dataView)

    let anomalies =
        ctx
            .Data
            .CreateEnumerable<PricePrediction>(transformedAnomalyData, reuseRowObject = false)
    let changePoints =
        ctx
            .Data
            .CreateEnumerable<PricePrediction>(transformedChangePointData, reuseRowObject = false)

    let priceChartData =
        anomalies
        |> Seq.map (fun p -> let p' = float (p.Prediction).[1] in (p.Date, p'))
        |> List.ofSeq
    
    let anomalyChartData =
        anomalies
        |> Seq.map (fun p -> let p' = if (p.Prediction).[0] = 0. then None else Some (float (p.Prediction).[1]) in (p.Date, p'))
        |> Seq.filter (fun (x,y) -> y.IsSome)
        |> Seq.map (fun (x,y) -> (x, y.Value))
        |> List.ofSeq

    let changePointChartData =
        changePoints
        |> Seq.map (fun p -> let p' = if (p.Prediction).[0] = 0. then None else Some (float (p.Prediction).[1]) in (p.Date, p'))
        |> Seq.filter (fun (x,y) -> y.IsSome)
        |> Seq.map (fun (x,y) -> (x, y.Value))
        |> List.ofSeq

    // Show Chart
    [priceChartData; anomalyChartData; changePointChartData]
    |> Chart.Combo
    |> Chart.WithOptions
        (Options(title = "Stock Price Anomalies and Inflection Points",
                        series = [| Series("lines"); Series("scatter"); Series("scatter") |],
                        displayAnnotations = true))
    |> Chart.WithLabels ["Price"; "Anomaly"; "ChangePoint"]
    |> Chart.WithLegend true
    |> Chart.WithSize (800, 400)
    |> Chart.Show

    
    let fileName = Path.Combine(staticFilesPath, "graph.html")
    //chart.Show() // Save the chart as an HTML file
    fileName
runPredictionPipeline @"C:\Users\Manali\Downloads\apptest\apptest\NFLX.csv"

// API Endpoints
//let handleUploadFile: WebPart = request (fun req ->
//    let filePath = @"C:\Users\Manali\Downloads\apptest\apptest\NFLX.csv"
//    try
//        runPredictionPipeline filePath
//        OK "File processed successfully."
//    with
//    | ex -> INTERNAL_ERROR ex.Message
//)

// Path to the file you want to serve
let filePath = @"C:\Users\Manali\AppData\Local\Temp\047b1965-683e-4387-a8c2-6173df9849de.html"

let handleUploadFile: WebPart =
    request (fun req ->
        match req.files |> Seq.tryHead with
        | Some file ->
            try
                // Directly process the uploaded file
                runPredictionPipeline file.tempFilePath
                OK $"File '{file.fileName}' processed successfully."
            with ex ->
                INTERNAL_ERROR $"Failed to process file: {ex.Message}"
        | None ->
            BAD_REQUEST "No file uploaded."
    )

let getGraph: WebPart =
    request (fun _ ->
        let filePath = @"C:\Users\Manali\Downloads\apptest\apptest\NFLX.csv"
        try
            let graphPath = @"file:///C:/Users/Manali/AppData/Local/Temp/047b1965-683e-4387-a8c2-6173df9849de.html"
            Files.file graphPath // Serve the generated graph file
        with ex ->
            INTERNAL_ERROR $"Failed to generate graph: {ex.Message}"
    )

let getMetrics: WebPart =
    request (fun _ ->
        match predictionsResult with
        | Some (mse, r2, _) -> OK (sprintf """{ "mse": %f, "r2": %f }""" mse r2)
        | None -> BAD_REQUEST "No data processed yet."
    )

let getPredictions: WebPart =
    request (fun _ ->
        match predictionsResult with
        | Some (_, _, (actual, predicted)) ->
            let json = Array.zip actual predicted
                         |> Array.map (fun (a, p) -> sprintf """{ "actual": %f, "predicted": %f }""" a p)
                         |> String.concat ","
            OK (sprintf """[ %s ]""" json)
        | None -> BAD_REQUEST "No data processed yet."
    )

let getNextPrice: WebPart =
    request (fun _ ->
        match nextPriceResult with
        | Some (nextPrice, nextDate) -> OK (sprintf """{ "nextPrice": %f, "nextDate": "%s" }""" nextPrice nextDate)
        | None -> BAD_REQUEST "No data processed yet."
    )

// Start the Suave server
let dashboardPath = @"C:\path\to\dashboard.html" // Path to your HTML file
//let filePath = @"C:\Users\Manali\AppData\Local\Temp\047b1965-683e-4387-a8c2-6173df9849de.html" // Path to your file

let app =
    choose [
        
        //path "/file" >=> file filePath // Serve the file
        //GET >=> path "/" >=> OK "<h2>Access Your File</h2><a href='/file'>Open Graph File</a>" // Main dashboard
        GET >=> path "/" >=> file dashboardPath // Serve the main dashboard HTML file
        GET >=> path "/file" >=> file filePath // Serve the graph file
        POST >=> path "/upload" >=> handleUploadFile
        GET >=> path "/graph" >=> getGraph
        GET >=> path "/metrics" >=> getMetrics
        GET >=> path "/predictions" >=> getPredictions
        GET >=> path "/next-price" >=> getNextPrice
        GET >=> path "/" >=> Files.file (Path.Combine(staticFilesPath, "index.html")) // Correctly serve index.html
        GET >=> pathScan "/StaticFiles/%s" (fun fileName ->
            Files.file (Path.Combine(staticFilesPath, fileName)) // Serve all files in StaticFiles folder
        )
    ]

let config = { defaultConfig with bindings = [ HttpBinding.createSimple HTTP "127.0.0.1" 8080 ] }
startWebServer defaultConfig app
