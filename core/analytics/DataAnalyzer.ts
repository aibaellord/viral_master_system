import { Injectable } from '@nestjs/common';
import { DataProcessor } from '../data/DataProcessor';
import { MLEngine } from '../ml/MLEngine';
import { Visualizer } from '../visualization/Visualizer';

@Injectable()
export class DataAnalyzer {
constructor(
    private dataProcessor: DataProcessor,
    private mlEngine: MLEngine,
    private visualizer: Visualizer
) {}

// Data Processing
async processData(data: RawData): Promise<ProcessedData> {
    const cleanedData = await this.dataProcessor.cleanData(data);
    const normalizedData = await this.dataProcessor.normalizeData(cleanedData);
    return this.dataProcessor.transformData(normalizedData);
}

// Pattern Recognition
async recognizePatterns(data: ProcessedData): Promise<Pattern[]> {
    const features = await this.extractFeatures(data);
    const patterns = await this.mlEngine.detectPatterns(features);
    return this.validatePatterns(patterns);
}

// Trend Analysis
async analyzeTrends(data: ProcessedData): Promise<TrendAnalysis> {
    const timeSeriesData = await this.prepareTimeSeries(data);
    const trends = await this.mlEngine.analyzeTrends(timeSeriesData);
    return this.generateTrendReport(trends);
}

// Predictive Modeling
async createPredictiveModel(data: ProcessedData): Promise<PredictiveModel> {
    const trainedModel = await this.mlEngine.trainModel(data);
    await this.validateModel(trainedModel);
    return this.optimizeModel(trainedModel);
}

// Statistical Analysis
async performStatistics(data: ProcessedData): Promise<StatisticalResults> {
    const stats = await this.calculateStatistics(data);
    await this.validateStatistics(stats);
    return this.generateStatsReport(stats);
}

// Machine Learning
async applyML(data: ProcessedData): Promise<MLResults> {
    const features = await this.extractFeatures(data);
    const predictions = await this.mlEngine.makePredictions(features);
    return this.validatePredictions(predictions);
}

// Anomaly Detection
async detectAnomalies(data: ProcessedData): Promise<Anomaly[]> {
    const normalizedData = await this.normalizeData(data);
    const anomalies = await this.mlEngine.findAnomalies(normalizedData);
    return this.categorizeAnomalies(anomalies);
}

// Data Visualization
async createVisualizations(data: ProcessedData): Promise<Visualization[]> {
    const preparedData = await this.prepareForVisualization(data);
    const visuals = await this.visualizer.generateVisuals(preparedData);
    return this.optimizeVisuals(visuals);
}

// Performance Metrics
async calculateMetrics(data: ProcessedData): Promise<PerformanceMetrics> {
    const metrics = await this.computeMetrics(data);
    await this.validateMetrics(metrics);
    return this.generateMetricsReport(metrics);
}
}

