module MyApp.Program

open System
open Deedle
open MathNet.Numerics.LinearAlgebra
open Accord.Statistics.Models.Regression.Linear
open Suave
open Suave.Filters
open Suave.Operators
open Suave.Successful
open Suave.RequestErrors


// Load and preprocess data
let loadData (filePath: string) =
    let df = Frame.ReadCsv(filePath)
    let requiredColumns = ["Date"; "Open"; "High"; "Low"; "Close"; "Volume"; "Adj Close"]
    if not (requiredColumns |> List.forall (fun col -> df.ColumnKeys |> Seq.contains col)) then
        failwith "Required columns not found in the dataset."
    df.DropSparseRows()

let createFeaturesAndLabels (df: Frame<int, string>) =
    // Use the adjusted close price as the label
    let labels = df.GetColumn<float>("Adj Close") |> Series.values |> Array.ofSeq

    // Create lagged features (e.g., prices from the previous days)
    let laggedFeatures =
        [1 .. 5]
        |> List.map (fun lag -> df.GetColumn<float>("Adj Close") |> Series.shift -lag |> Series.values |> Array.ofSeq)
        |> Array.ofList

    // Find the minimum length of all lagged features to avoid mismatch
    let minLength = laggedFeatures |> Array.minBy Array.length |> Array.length

    // Trim all lagged features to the same length
    let trimmedLaggedFeatures = laggedFeatures |> Array.map (fun feature -> feature.[0..minLength - 1])

    // Now trim the labels to the same length
    let trimmedLabels = labels.[5..minLength + 4]  // Adjust for the 5-day lag

    // Transpose and filter out invalid rows with NaN
    let features = Array.transpose trimmedLaggedFeatures
    let validData =
        Array.zip features trimmedLabels
        |> Array.filter (fun (f, l) -> not (Array.exists Double.IsNaN f || Double.IsNaN l))

    Array.unzip validData

let splitData (features: float[][]) (labels: float[]) =
    let n = features.Length
    let trainSize = int (0.2 * float n)
    let trainX, testX = Array.splitAt trainSize features
    let trainY, testY = Array.splitAt trainSize labels
    (trainX, trainY, testX, testY)

let trainLinearRegression (trainX: float[][]) (trainY: float[]) =
    let model = MultipleLinearRegression(trainX.[0].Length)
    model.Regress(trainX, trainY)
    model

// Manual implementation of performance metrics
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
    (mse, r2, predictions)

let runPredictionPipeline filePath =
    let data = loadData filePath
    let features, labels = createFeaturesAndLabels data
    let trainX, trainY, testX, testY = splitData features labels

    // Train and evaluate linear regression

    let lrModel = trainLinearRegression trainX trainY
    let lrMse, lrR2, predictions = evaluateLinearRegression lrModel testX testY

    // Print results
    printfn "Linear Regression - MSE: %f, R2: %f" lrMse lrR2

    // Print some actual vs predicted values
    Array.zip testY predictions
    |> Array.take 10
    |> Array.iteri (fun i (actual, predicted) -> 
        printfn "Sample %d - Actual: $%.2f, Predicted: $%.2f" i actual predicted)

    // Predict next price
    let lastFeatures = features |> Array.last
    let nextPrice = lrModel.Transform(lastFeatures)
    
    // Get the last known date from the dataset
    let lastDate = data.GetColumn<DateTime>("Date").Values |> Seq.last
    let nextDate = lastDate.AddDays(1.0)

    printfn "Next predicted price for %s: $%.2f" (nextDate.ToString("yyyy-MM-dd")) nextPrice


// Run the pipeline
runPredictionPipeline @"C:\Users\Manali\Downloads\apptest\apptest\NFLX.csv"