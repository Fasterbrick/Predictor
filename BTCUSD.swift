import SwiftUI
import CoreML


struct BTCUSD: View {
    @State private var openPriceInput: String = ""
    
    // Prediction Result States
    @State private var m1CloseResult: String = ""
    @State private var m1HighResult: String = ""
    @State private var m1LowResult: String = ""
    @State private var m5CloseResult: String = ""
    @State private var m5HighResult: String = ""
    @State private var m5LowResult: String = ""
    @State private var m15CloseResult: String = ""
    @State private var m15HighResult: String = ""
    @State private var m15LowResult: String = ""
    @State private var m30CloseResult: String = ""
    @State private var m30HighResult: String = ""
    @State private var m30LowResult: String = ""
    @State private var H1CloseResult: String = ""
    @State private var H1HighResult: String = ""
    @State private var H1LowResult: String = ""
    @State private var H2CloseResult: String = ""
    @State private var H2HighResult: String = ""
    @State private var H2LowResult: String = ""
    @State private var H4CloseResult: String = ""
    @State private var H4HighResult: String = ""
    @State private var H4LowResult: String = ""
    @State private var DCloseResult: String = ""
    @State private var DHighResult: String = ""
    @State private var DLowResult: String = ""
    @State private var WCloseResult: String = ""
    @State private var WHighResult: String = ""
    @State private var WLowResult: String = ""
    @State private var MCloseResult: String = ""
    @State private var MHighResult: String = ""
    @State private var MLowResult: String = ""
    
    @State private var m1CloseDiff: String = ""
    @State private var m1HighDiff: String = ""
    @State private var m1LowDiff: String = ""
    @State private var m5CloseDiff: String = ""
    @State private var m5HighDiff: String = ""
    @State private var m5LowDiff: String = ""
    @State private var m15CloseDiff: String = ""
    @State private var m15HighDiff: String = ""
    @State private var m15LowDiff: String = ""
    @State private var m30CloseDiff: String = ""
    @State private var m30HighDiff: String = ""
    @State private var m30LowDiff: String = ""
    @State private var H1CloseDiff: String = ""
    @State private var H1HighDiff: String = ""
    @State private var H1LowDiff: String = ""
    @State private var H2CloseDiff: String = ""
    @State private var H2HighDiff: String = ""
    @State private var H2LowDiff: String = ""
    @State private var H4CloseDiff: String = ""
    @State private var H4HighDiff: String = ""
    @State private var H4LowDiff: String = ""
    @State private var DCloseDiff: String = ""
    @State private var DHighDiff: String = ""
    @State private var DLowDiff: String = ""
    @State private var WCloseDiff: String = ""
    @State private var WHighDiff: String = ""
    @State private var WLowDiff: String = ""
    @State private var MCloseDiff: String = ""
    @State private var MHighDiff: String = ""
    @State private var MLowDiff: String = ""
    
    @State private var currentDate = Date() // State variable for current date
    @State private var isPressed = false

    
    // Function to process the Open Price and generate predictions
    private func loadAndPredict() {
        guard let openPrice = Double(openPriceInput) else {
            print("Invalid input: Please enter a valid numeric value for Open Price.")
            resetResults()
            return
        }
        
        let openPrices = ["manual_input": openPrice]
        runModels(withOpenPrices: openPrices)
    }


    
    private func runModels(withOpenPrices openPrices: [String: Double]) {
        for (_, openPrice) in openPrices {
            let inputFeatures1mClose = m1BTCUSDCloseInput(_OPEN_: openPrice)
            let inputFeatures1mHigh = m1BTCUSDHighInput(_OPEN_: openPrice)
            let inputFeatures1mLow = m1BTCUSDLowInput(_OPEN_: openPrice)
            
            let inputFeatures5mClose = m5BTCUSDCloseInput(_OPEN_: openPrice)
            let inputFeatures5mHigh = m5BTCUSDHighInput(_OPEN_: openPrice)
            let inputFeatures5mLow = m5BTCUSDLowInput(_OPEN_: openPrice)
            
            let inputFeatures15mClose = m15BTCUSDCloseInput(_OPEN_: openPrice)
            let inputFeatures15mHigh = m15BTCUSDHighInput(_OPEN_: openPrice)
            let inputFeatures15mLow = m15BTCUSDLowInput(_OPEN_: openPrice)
            
            let inputFeatures30mClose = m30BTCUSDCloseInput(_OPEN_: openPrice)
            let inputFeatures30mHigh = m30BTCUSDHighInput(_OPEN_: openPrice)
            let inputFeatures30mLow = m30BTCUSDLowInput(_OPEN_: openPrice)
            
            let inputFeatures1HClose = H1BTCUSDCloseInput(_OPEN_: openPrice)
            let inputFeatures1HHigh = H1BTCUSDHighInput(_OPEN_: openPrice)
            let inputFeatures1HLow = H1BTCUSDLowInput(_OPEN_: openPrice)
            
            let inputFeatures2HClose = H2BTCUSDCloseInput(_OPEN_: openPrice)
            let inputFeatures2HHigh = H2BTCUSDHighInput(_OPEN_: openPrice)
            let inputFeatures2HLow = H2BTCUSDLowInput(_OPEN_: openPrice)
            
            let inputFeatures4HClose = H4BTCUSDCloseInput(_OPEN_: openPrice)
            let inputFeatures4HHigh = H4BTCUSDHighInput(_OPEN_: openPrice)
            let inputFeatures4HLow = H4BTCUSDLowInput(_OPEN_: openPrice)
            
            let inputFeaturesDailyClose = DBTCUSDCloseInput(OPEN: openPrice)
            let inputFeaturesDailyHigh = DBTCUSDHighInput(OPEN: openPrice)
            let inputFeaturesDailyLow = DBTCUSDLowInput(OPEN: openPrice)
            
            let inputFeaturesWeeklyClose = WBTCUSDCloseInput(OPEN: openPrice)
            let inputFeaturesWeeklyHigh = WBTCUSDHighInput(OPEN: openPrice)
            let inputFeaturesWeeklyLow = WBTCUSDLowInput(OPEN: openPrice)
            
            let inputFeaturesMonthlyClose = MBTCUSDCloseInput(OPEN: openPrice)
            let inputFeaturesMonthlyHigh = MBTCUSDHighInput(OPEN: openPrice)
            let inputFeaturesMonthlyLow = MBTCUSDLowInput(OPEN: openPrice)
            
            performPrediction(
                with: inputFeatures1mClose,
                inputFeatures1mHigh: inputFeatures1mHigh,
                inputFeatures1mLow: inputFeatures1mLow,
                
                inputFeatures5mClose: inputFeatures5mClose,
                inputFeatures5mHigh: inputFeatures5mHigh,
                inputFeatures5mLow: inputFeatures5mLow,
                
                
                inputFeatures15mClose: inputFeatures15mClose,
                inputFeatures15mHigh: inputFeatures15mHigh,
                inputFeatures15mLow: inputFeatures15mLow,
                
                
                inputFeatures30mClose: inputFeatures30mClose,
                inputFeatures30mHigh: inputFeatures30mHigh,
                inputFeatures30mLow: inputFeatures30mLow,
                
                
                inputFeatures1HClose: inputFeatures1HClose,
                inputFeatures1HHigh: inputFeatures1HHigh,
                inputFeatures1HLow: inputFeatures1HLow,
                
                inputFeatures2HClose: inputFeatures2HClose,
                inputFeatures2HHigh: inputFeatures2HHigh,
                inputFeatures2HLow: inputFeatures2HLow,
                
                inputFeatures4HClose: inputFeatures4HClose,
                inputFeatures4HHigh: inputFeatures4HHigh,
                inputFeatures4HLow: inputFeatures4HLow,
                
                inputFeaturesDailyClose: inputFeaturesDailyClose,
                inputFeaturesDailyHigh: inputFeaturesDailyHigh,
                inputFeaturesDailyLow: inputFeaturesDailyLow,
                
                inputFeaturesWeeklyClose: inputFeaturesWeeklyClose,
                inputFeaturesWeeklyHigh: inputFeaturesWeeklyHigh,
                inputFeaturesWeeklyLow: inputFeaturesWeeklyLow,
                
                inputFeaturesMonthlyClose: inputFeaturesMonthlyClose,
                inputFeaturesMonthlyHigh: inputFeaturesMonthlyHigh,
                inputFeaturesMonthlyLow: inputFeaturesMonthlyLow
            )
            
        }
    }

    
    private func performPrediction(
        with inputFeatures1mClose: m1BTCUSDCloseInput,
        inputFeatures1mHigh: m1BTCUSDHighInput,
        inputFeatures1mLow: m1BTCUSDLowInput,
        
        inputFeatures5mClose: m5BTCUSDCloseInput,
        inputFeatures5mHigh: m5BTCUSDHighInput,
        inputFeatures5mLow: m5BTCUSDLowInput,
        
        inputFeatures15mClose: m15BTCUSDCloseInput,
        inputFeatures15mHigh: m15BTCUSDHighInput,
        inputFeatures15mLow: m15BTCUSDLowInput,
        
        inputFeatures30mClose: m30BTCUSDCloseInput,
        inputFeatures30mHigh: m30BTCUSDHighInput,
        inputFeatures30mLow: m30BTCUSDLowInput,
        
        inputFeatures1HClose: H1BTCUSDCloseInput,
        inputFeatures1HHigh: H1BTCUSDHighInput,
        inputFeatures1HLow: H1BTCUSDLowInput,
        
        inputFeatures2HClose: H2BTCUSDCloseInput,
        inputFeatures2HHigh: H2BTCUSDHighInput,
        inputFeatures2HLow: H2BTCUSDLowInput,
        
        inputFeatures4HClose: H4BTCUSDCloseInput,
        inputFeatures4HHigh: H4BTCUSDHighInput,
        inputFeatures4HLow: H4BTCUSDLowInput,
        
        inputFeaturesDailyClose: DBTCUSDCloseInput,
        inputFeaturesDailyHigh: DBTCUSDHighInput,
        inputFeaturesDailyLow: DBTCUSDLowInput,
        
        inputFeaturesWeeklyClose: WBTCUSDCloseInput,
        inputFeaturesWeeklyHigh: WBTCUSDHighInput,
        inputFeaturesWeeklyLow: WBTCUSDLowInput,
        
        inputFeaturesMonthlyClose: MBTCUSDCloseInput,
        inputFeaturesMonthlyHigh: MBTCUSDHighInput,
        inputFeaturesMonthlyLow: MBTCUSDLowInput
    ) {
        do {
            guard let openPrice = Double(openPriceInput) else {
                print("Invalid Open Price")
                resetResults()
                return
            }
            
            // Load the models with optimized configuration
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            
            // Load the models
            let m1CloseModel = try m1BTCUSDClose(configuration: configuration)
            let m1HighModel = try m1BTCUSDHigh(configuration: configuration)
            let m1LowModel = try m1BTCUSDLow(configuration: configuration)
            
            let m5CloseModel = try m5BTCUSDClose(configuration: configuration)
            let m5HighModel = try m5BTCUSDHigh(configuration: configuration)
            let m5LowModel = try m5BTCUSDLow(configuration: configuration)
            
            let m15CloseModel = try m15BTCUSDClose(configuration: configuration)
            let m15HighModel = try m15BTCUSDHigh(configuration: configuration)
            let m15LowModel = try m15BTCUSDLow(configuration: configuration)
            
            let m30CloseModel = try m30BTCUSDClose(configuration: configuration)
            let m30HighModel = try m30BTCUSDHigh(configuration: configuration)
            let m30LowModel = try m30BTCUSDLow(configuration: configuration)
            
            let H1CloseModel = try H1BTCUSDClose(configuration: configuration)
            let H1HighModel = try H1BTCUSDHigh(configuration: configuration)
            let H1LowModel = try H1BTCUSDLow(configuration: configuration)
            
            let H2CloseModel = try H2BTCUSDClose(configuration: configuration)
            let H2HighModel = try H2BTCUSDHigh(configuration: configuration)
            let H2LowModel = try H2BTCUSDLow(configuration: configuration)
            
            let H4CloseModel = try H4BTCUSDClose(configuration: configuration)
            let H4HighModel = try H4BTCUSDHigh(configuration: configuration)
            let H4LowModel = try H4BTCUSDLow(configuration: configuration)
            
            let DCloseModel = try DBTCUSDClose(configuration: configuration)
            let DHighModel = try DBTCUSDHigh(configuration: configuration)
            let DLowModel = try DBTCUSDLow(configuration: configuration)
            
            let WCloseModel = try WBTCUSDClose(configuration: configuration)
            let WHighModel = try WBTCUSDHigh(configuration: configuration)
            let WLowModel = try WBTCUSDLow(configuration: configuration)
            
            let MCloseModel = try MBTCUSDClose(configuration: configuration)
            let MHighModel = try MBTCUSDHigh(configuration: configuration)
            let MLowModel = try MBTCUSDLow(configuration: configuration)
            
            // Perform predictions for 1m
            
            let m1CloseOutput = try m1CloseModel.prediction(input: inputFeatures1mClose)
            let m1HighOutput = try m1HighModel.prediction(input: inputFeatures1mHigh)
            let m1LowOutput = try m1LowModel.prediction(input: inputFeatures1mLow)
            
            m1CloseResult = formatPrediction(m1CloseOutput._CLOSE_)
            m1HighResult = formatPrediction(m1HighOutput._HIGH_)
            m1LowResult = formatPrediction(m1LowOutput._LOW_)
            
            let m1CloseDiffValue = calculateDifference(predictedValue: m1CloseOutput._CLOSE_, openPrice: openPrice)
            m1CloseDiff = formatPips(m1CloseDiffValue)
            
            let m1HighDiffValue = calculateDifference(predictedValue: m1HighOutput._HIGH_, openPrice: openPrice)
            m1HighDiff = formatPips(m1HighDiffValue)
            
            let m1LowDiffValue = calculateDifference(predictedValue: m1LowOutput._LOW_, openPrice: openPrice)
            m1LowDiff = formatPips(m1LowDiffValue)
            
            // Perform predictions for 5m
                        let m5CloseOutput = try m5CloseModel.prediction(input: inputFeatures5mClose)
                        let m5HighOutput = try m5HighModel.prediction(input: inputFeatures5mHigh)
                        let m5LowOutput = try m5LowModel.prediction(input: inputFeatures5mLow)
                        
                        m5CloseResult = formatPrediction(m5CloseOutput._CLOSE_)
                        m5HighResult = formatPrediction(m5HighOutput._HIGH_)
                        m5LowResult = formatPrediction(m5LowOutput._LOW_)
                        
                        let m5CloseDiffValue = calculateDifference(predictedValue: m5CloseOutput._CLOSE_, openPrice: openPrice)
                        m5CloseDiff = formatPips(m5CloseDiffValue)
                        
                        let m5HighDiffValue = calculateDifference(predictedValue: m5HighOutput._HIGH_, openPrice: openPrice)
                        m5HighDiff = formatPips(m5HighDiffValue)
                        
                        let m5LowDiffValue = calculateDifference(predictedValue: m5LowOutput._LOW_, openPrice: openPrice)
                        m5LowDiff = formatPips(m5LowDiffValue)
                        
                        // Perform predictions for 15m
                        let m15CloseOutput = try m15CloseModel.prediction(input: inputFeatures15mClose)
                        let m15HighOutput = try m15HighModel.prediction(input: inputFeatures15mHigh)
                        let m15LowOutput = try m15LowModel.prediction(input: inputFeatures15mLow)
                        
                        m15CloseResult = formatPrediction(m15CloseOutput._CLOSE_)
                        m15HighResult = formatPrediction(m15HighOutput._HIGH_)
                        m15LowResult = formatPrediction(m15LowOutput._LOW_)
                        
                        let m15CloseDiffValue = calculateDifference(predictedValue: m15CloseOutput._CLOSE_, openPrice: openPrice)
                        m15CloseDiff = formatPips(m15CloseDiffValue)
                        
                        let m15HighDiffValue = calculateDifference(predictedValue: m15HighOutput._HIGH_, openPrice: openPrice)
                        m15HighDiff = formatPips(m15HighDiffValue)
                        
                        let m15LowDiffValue = calculateDifference(predictedValue: m15LowOutput._LOW_, openPrice: openPrice)
                        m15LowDiff = formatPips(m15LowDiffValue)
                        
                        // Perform predictions for 30m
                        let m30CloseOutput = try m30CloseModel.prediction(input: inputFeatures30mClose)
                        let m30HighOutput = try m30HighModel.prediction(input: inputFeatures30mHigh)
                        let m30LowOutput = try m30LowModel.prediction(input: inputFeatures30mLow)
                        
                        m30CloseResult = formatPrediction(m30CloseOutput._CLOSE_)
                        m30HighResult = formatPrediction(m30HighOutput._HIGH_)
                        m30LowResult = formatPrediction(m30LowOutput._LOW_)
                        
                        let m30CloseDiffValue = calculateDifference(predictedValue: m30CloseOutput._CLOSE_, openPrice: openPrice)
                        m30CloseDiff = formatPips(m30CloseDiffValue)
                        
                        let m30HighDiffValue = calculateDifference(predictedValue: m30HighOutput._HIGH_, openPrice: openPrice)
                        m30HighDiff = formatPips(m30HighDiffValue)
                        
                        let m30LowDiffValue = calculateDifference(predictedValue: m30LowOutput._LOW_, openPrice: openPrice)
                        m30LowDiff = formatPips(m30LowDiffValue)
                        
                        
                        // Perform predictions for 1H
                        let H1CloseOutput = try H1CloseModel.prediction(input: inputFeatures1HClose)
                        let H1HighOutput = try H1HighModel.prediction(input: inputFeatures1HHigh)
                        let H1LowOutput = try H1LowModel.prediction(input: inputFeatures1HLow)
                        
                        H1CloseResult = formatPrediction(H1CloseOutput._CLOSE_)
                        H1HighResult = formatPrediction(H1HighOutput._HIGH_)
                        H1LowResult = formatPrediction(H1LowOutput._LOW_)
                        
                        let H1CloseDiffValue = calculateDifference(predictedValue: H1CloseOutput._CLOSE_, openPrice: openPrice)
                        H1CloseDiff = formatPips(H1CloseDiffValue)
                        
                        let H1HighDiffValue = calculateDifference(predictedValue: H1HighOutput._HIGH_, openPrice: openPrice)
                        H1HighDiff = formatPips(H1HighDiffValue)
                        
                        let H1LowDiffValue = calculateDifference(predictedValue: H1LowOutput._LOW_, openPrice: openPrice)
                        H1LowDiff = formatPips(H1LowDiffValue)
                        
                        // Perform predictions for 2H
                        let H2CloseOutput = try H2CloseModel.prediction(input: inputFeatures2HClose)
                        let H2HighOutput = try H2HighModel.prediction(input: inputFeatures2HHigh)
                        let H2LowOutput = try H2LowModel.prediction(input: inputFeatures2HLow)
                        
                        H2CloseResult = formatPrediction(H2CloseOutput._CLOSE_)
                        H2HighResult = formatPrediction(H2HighOutput._HIGH_)
                        H2LowResult = formatPrediction(H2LowOutput._LOW_)
                        
                        let H2CloseDiffValue = calculateDifference(predictedValue: H2CloseOutput._CLOSE_, openPrice: openPrice)
                        H2CloseDiff = formatPips(H2CloseDiffValue)
                        
                        let H2HighDiffValue = calculateDifference(predictedValue: H2HighOutput._HIGH_, openPrice: openPrice)
                        H2HighDiff = formatPips(H2HighDiffValue)
                        
                        let H2LowDiffValue = calculateDifference(predictedValue: H2LowOutput._LOW_, openPrice: openPrice)
                        H2LowDiff = formatPips(H2LowDiffValue)
                        
                        // Perform predictions for 4H
                        let H4CloseOutput = try H4CloseModel.prediction(input: inputFeatures4HClose)
                        let H4HighOutput = try H4HighModel.prediction(input: inputFeatures4HHigh)
                        let H4LowOutput = try H4LowModel.prediction(input: inputFeatures4HLow)
                        
                        H4CloseResult = formatPrediction(H4CloseOutput._CLOSE_)
                        H4HighResult = formatPrediction(H4HighOutput._HIGH_)
                        H4LowResult = formatPrediction(H4LowOutput._LOW_)
                        
                        let H4CloseDiffValue = calculateDifference(predictedValue: H4CloseOutput._CLOSE_, openPrice: openPrice)
                        H4CloseDiff = formatPips(H4CloseDiffValue)
                        
                        let H4HighDiffValue = calculateDifference(predictedValue: H4HighOutput._HIGH_, openPrice: openPrice)
                        H4HighDiff = formatPips(H4HighDiffValue)
                        
                        let H4LowDiffValue = calculateDifference(predictedValue: H4LowOutput._LOW_, openPrice: openPrice)
                        H4LowDiff = formatPips(H4LowDiffValue)
                        
                        // Perform predictions for Daily
                        let DCloseOutput = try DCloseModel.prediction(input: inputFeaturesDailyClose)
                        let DHighOutput = try DHighModel.prediction(input: inputFeaturesDailyHigh)
                        let DLowOutput = try DLowModel.prediction(input: inputFeaturesDailyLow)
                        

                                    DCloseResult = formatPrediction(DCloseOutput.CLOSE)
                                    DHighResult = formatPrediction(DHighOutput.HIGH)
                                    DLowResult = formatPrediction(DLowOutput.LOW)
                                    
                                    let DCloseDiffValue = calculateDifference(predictedValue: DCloseOutput.CLOSE, openPrice: openPrice)
                                    DCloseDiff = formatPips(DCloseDiffValue)
                                    
                                    let DHighDiffValue = calculateDifference(predictedValue: DHighOutput.HIGH, openPrice: openPrice)
                                    DHighDiff = formatPips(DHighDiffValue)
                                    
                                    let DLowDiffValue = calculateDifference(predictedValue: DLowOutput.LOW, openPrice: openPrice)
                                    DLowDiff = formatPips(DLowDiffValue)
                                    
                                    // Perform predictions for Weekly
                                    let WCloseOutput = try WCloseModel.prediction(input: inputFeaturesWeeklyClose)
                                    let WHighOutput = try WHighModel.prediction(input: inputFeaturesWeeklyHigh)
                                    let WLowOutput = try WLowModel.prediction(input: inputFeaturesWeeklyLow)
                                    
                                    WCloseResult = formatPrediction(WCloseOutput.CLOSE)
                                    WHighResult = formatPrediction(WHighOutput.HIGH)
                                    WLowResult = formatPrediction(WLowOutput.LOW)
                                    
                                    let WCloseDiffValue = calculateDifference(predictedValue: WCloseOutput.CLOSE, openPrice: openPrice)
                                    WCloseDiff = formatPips(WCloseDiffValue)
                                    
                                    let WHighDiffValue = calculateDifference(predictedValue: WHighOutput.HIGH, openPrice: openPrice)
                                    WHighDiff = formatPips(WHighDiffValue) // Format and store the result

                                    
                                    let WLowDiffValue = calculateDifference(predictedValue: WLowOutput.LOW, openPrice: openPrice)
                                    WLowDiff = formatPips(WLowDiffValue)
                                    
                                    // Perform predictions for Monthly
                                    let MCloseOutput = try MCloseModel.prediction(input: inputFeaturesMonthlyClose)
                                    let MHighOutput = try MHighModel.prediction(input: inputFeaturesMonthlyHigh)
                                    let MLowOutput = try MLowModel.prediction(input: inputFeaturesMonthlyLow)
                                    
                                    MCloseResult = formatPrediction(MCloseOutput.CLOSE)
                                    MHighResult = formatPrediction(MHighOutput.HIGH)
                                    MLowResult = formatPrediction(MLowOutput.LOW)
                                    
                                    // difference
                                    let MCloseDiffValue = calculateDifference(predictedValue: MCloseOutput.CLOSE, openPrice: openPrice)
                                    
                                    MCloseDiff = formatPips(MCloseDiffValue)
                                    
                                    let MHighDiffValue = calculateDifference(predictedValue: MHighOutput.HIGH, openPrice: openPrice)
                                   
                                    MHighDiff = formatPips(MHighDiffValue)
                                    
                                    let MLowDiffValue = calculateDifference(predictedValue: MLowOutput.LOW, openPrice: openPrice)
                                    
                                    MLowDiff = formatPips(MLowDiffValue)
            
        } catch {
            print("Prediction failed: \(error.localizedDescription)")
            resetResults()
        }
    }
    
    private func calculateDifference(predictedValue: Double, openPrice: Double) -> Double {
        return predictedValue - openPrice
    }
    
    private func formatPrediction(_ value: Double) -> String {
        return String(format: "%.2f", value)
    }
    
    private func formatPips(_ value: Double) -> String {
        
        let pips = value * 1
        // Return the formatted value as a string with 1 decimal place
        return String(format: "%.2f", pips)
    }

    private func resetResults() {
        m1CloseResult = ""
        m1HighResult = ""
        m1LowResult = ""
        m5CloseResult = ""
        m5HighResult = ""
        m5LowResult = ""
        m15CloseResult = ""
        m15HighResult = ""
        m15LowResult = ""
        m30CloseResult = ""
        m30HighResult = ""
        m30LowResult = ""
        H1CloseResult = ""
        H1HighResult = ""
        H1LowResult = ""
        H2CloseResult = ""
        H2HighResult = ""
        H2LowResult = ""
        H4CloseResult = ""
        H4HighResult = ""
        H4LowResult = ""
        DCloseResult = ""
        DHighResult = ""
        DLowResult = ""
        WCloseResult = ""
        WHighResult = ""
        WLowResult = ""
        MCloseResult = ""
        MHighResult = ""
        MLowResult = ""
        
        m1CloseDiff = ""
        m1HighDiff = ""
        m1LowDiff = ""
        m5CloseDiff = ""
        m5HighDiff = ""
        m5LowDiff = ""
        m15CloseDiff = ""
        m15HighDiff = ""
        m15LowDiff = ""
        m30CloseDiff = ""
        m30HighDiff = ""
        m30LowDiff = ""
        H1CloseDiff = ""
        H1HighDiff = ""
        H1LowDiff = ""
        H2CloseDiff = ""
        H2HighDiff = ""
        H2LowDiff = ""
        H4CloseDiff = ""
        H4HighDiff = ""
        H4LowDiff = ""
        DCloseDiff = ""
        DHighDiff = ""
        DLowDiff = ""
        WCloseDiff = ""
        WHighDiff = ""
        WLowDiff = ""
        MCloseDiff = ""
        MHighDiff = ""
        MLowDiff = ""
        
    }

    func colorForDifference(_ diff: String) -> Color {
        guard let diffValue = Double(diff) else { return .gray }
        return diffValue < 0 ? .red : diffValue == 0 ? .gray : .green
    }

    var body: some View {
        ScrollView { // Added for scrollability
            VStack(spacing: 20) {
                // Header
                VStack {

                    Text("BTC/USD")
                        .foregroundStyle(.white)
                }
                .padding(.top, 40)
                
                // Input Field for Open Price
                VStack(alignment: .leading, spacing: 10) {
                    
                    HStack {
                        Text("Open Price:")
                            .foregroundColor(.white)
                        TextField("Enter Open Price", text: $openPriceInput)
//                            .keyboardType(.decimalPad)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .padding(5)
                            .background(Color.white.opacity(0.1))
                            .cornerRadius(5)
//                            .foregroundColor(.white)
                    }
                }
                .padding()
                .background(Color.black.opacity(0.3))
                .cornerRadius(10)
                
                // Predict Button
                Button(action: loadAndPredict) {
                    HStack(spacing: 15) {
                        Image(systemName: "magnifyingglass")
                            .font(.system(size: 16, weight: .bold))
                        Text("Predict")
                            .font(.system(size: 17, weight: .medium))
                    }
                    .padding(.vertical, 15)
                    .padding(.horizontal, 30)
                    .frame(maxWidth: .infinity)
                    .background(
                        ZStack {
                            LinearGradient(
                                gradient: Gradient(colors: [Color.white.opacity(0.15), Color.white.opacity(0.05)]),
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                            .blur(radius: 2)
                            RoundedRectangle(cornerRadius: 14)
                                .stroke(Color.white.opacity(0.2), lineWidth: 1)
                        }
                    )
                    .foregroundColor(.white)
                    .cornerRadius(14)
                    .shadow(color: Color.black.opacity(0.2), radius: 8, x: 0, y: 4)
                    .scaleEffect(isPressed ? 0.98 : 1)
                    .animation(.easeInOut(duration: 0.2), value: isPressed)
                }
                .buttonStyle(.plain)
                .padding(.horizontal)
                
                // Prediction Table
                if !m1CloseResult.isEmpty {
                    VStack {
                        HStack {
                            Text("Timeframe").fontWeight(.bold).frame(maxWidth: .infinity, alignment: .leading)
                            Text("Close").fontWeight(.bold).frame(maxWidth: .infinity, alignment: .center)
                            Text("🎯").fontWeight(.bold).frame(maxWidth: .infinity, alignment: .center)
                            Text("High").fontWeight(.bold).frame(maxWidth: .infinity, alignment: .center)
                            Text("⬆️").fontWeight(.bold).frame(maxWidth: .infinity, alignment: .center)
                            Text("Low").fontWeight(.bold).frame(maxWidth: .infinity, alignment: .center)
                            Text("⬇️").fontWeight(.bold).frame(maxWidth: .infinity, alignment: .center)
                        }
                        .foregroundColor(.white)
                        .padding(.bottom, 10)
                        
                        ForEach([
                            ("1m", m1CloseResult, m1HighResult, m1LowResult, m1CloseDiff, m1HighDiff, m1LowDiff),
                            ("5m", m5CloseResult, m5HighResult, m5LowResult, m5CloseDiff, m5HighDiff, m5LowDiff),
                            ("15m", m15CloseResult, m15HighResult, m15LowResult, m15CloseDiff, m15HighDiff, m15LowDiff),
                            ("30m", m30CloseResult, m30HighResult, m30LowResult, m30CloseDiff, m30HighDiff, m30LowDiff),
                            ("1H", H1CloseResult, H1HighResult, H1LowResult, H1CloseDiff, H1HighDiff, H1LowDiff),
                            ("2H", H2CloseResult, H2HighResult, H2LowResult, H2CloseDiff, H2HighDiff, H2LowDiff),
                            ("4H", H4CloseResult, H4HighResult, H4LowResult, H4CloseDiff, H4HighDiff, H4LowDiff),
                            ("Daily", DCloseResult, DHighResult, DLowResult, DCloseDiff, DHighDiff, DLowDiff),
                            ("Weekly", WCloseResult, WHighResult, WLowResult, WCloseDiff, WHighDiff, WLowDiff),
                            ("Monthly", MCloseResult, MHighResult, MLowResult, MCloseDiff, MHighDiff, MLowDiff)
                        ], id: \.0) { timeframe, close, high, low, closeDiff, highDiff, lowDiff in
                            HStack {
                                Text(timeframe).foregroundColor(.white).frame(maxWidth: .infinity, alignment: .leading)
                                Text(close).foregroundColor(.white).frame(maxWidth: .infinity, alignment: .center)
                                Text(closeDiff).foregroundColor(colorForDifference(closeDiff)).frame(maxWidth: .infinity, alignment: .center)
                                Text(high).foregroundColor(.white).frame(maxWidth: .infinity, alignment: .center)
                                Text(highDiff).foregroundColor(colorForDifference(highDiff)).frame(maxWidth: .infinity, alignment: .center)
                                Text(low).foregroundColor(.white).frame(maxWidth: .infinity, alignment: .center)
                                Text(lowDiff).foregroundColor(colorForDifference(lowDiff)).frame(maxWidth: .infinity, alignment: .center)
                            }
                            .padding(.vertical, 5)
                            .background(Color.black.opacity(0.1))
                        }
                    }
                    .padding()
                    .background(Color.black.opacity(0.2))
                    .cornerRadius(15)
                    .padding()
                }
                
                Spacer()
            }
            .padding()
        }
        .onReceive(Timer.publish(every: 1, on: .main, in: .common).autoconnect()) { _ in
            currentDate = Date()
        }
#if os(iOS)
        .onTapGesture { // Dismiss keyboard when tapping outside
            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
        }
#endif
        .background(
            LinearGradient(
                gradient: Gradient(colors: [
                    Color(red: 0.08, green: 0.12, blue: 0.24),
                    Color(red: 0.02, green: 0.24, blue: 0.31),
                    Color(red: 0.11, green: 0.35, blue: 0.41)
                ]),
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .edgesIgnoringSafeArea(.all)
    }
    

}
