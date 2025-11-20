
import React, { useState, useCallback } from 'react';
import { Header } from './components/Header';
import { ImageUploader } from './components/ImageUploader';
import { AnalysisDisplay } from './components/AnalysisDisplay';
import { Loader } from './components/Loader';
import { analyzeImage, generateAnalysisDescription } from './services/geminiService';
import { detectObjects } from './services/yoloService';
import type { DetectionResult } from './types';
import { IconAlertTriangle, IconSparkles } from './components/IconComponents';
import { AnalysisDescription } from './components/AnalysisDescription';

const App: React.FC = () => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [results, setResults] = useState<DetectionResult[] | null>(null);
  const [analysisDescription, setAnalysisDescription] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = useCallback(async (base64: string, file: File) => {
    setImageUrl(base64);
    setResults(null);
    setAnalysisDescription(null);
    setError(null);
    setIsLoading(true);

    try {
      // Create image element for YOLO detection
      const img = new Image();
      img.src = base64;
      await new Promise((resolve) => {
        img.onload = resolve;
      });

      // Run YOLO detection
      const yoloResults = await detectObjects(img);
      console.log('YOLO detections:', yoloResults);

      // Log detection source for debugging
      if (yoloResults.length > 0 && yoloResults[0].class === 'No Endodontic Treatment' && yoloResults[0].bbox[0] === 100) {
        console.log('⚠️ Using mock dental detections - ONNX model not loaded properly');
      }

      // Run Gemini analysis
      const analysisResults = await analyzeImage(base64.split(',')[1], file.type);
      setResults(analysisResults);

      if (analysisResults && analysisResults.length > 0) {
        const description = await generateAnalysisDescription(analysisResults, yoloResults);
        setAnalysisDescription(description);
      } else if (analysisResults) {
        setAnalysisDescription("No specific root canal issues were detected by the model. A comprehensive clinical examination is always recommended for a complete diagnosis.");
      }
    } catch (e) {
      console.error(e);
      setError('Failed to analyze the image. The AI model may be unable to process this image or an API error occurred. Please try another image.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleReset = () => {
    setImageUrl(null);
    setResults(null);
    setAnalysisDescription(null);
    setError(null);
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-gray-200 flex flex-col font-sans">
      <Header />
      <main className="flex-grow container mx-auto p-4 md:p-8 flex flex-col">
        <div className="flex-grow grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          <div className="w-full h-full flex flex-col">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">Upload Radiograph</h2>
            <div className="flex-grow">
              <ImageUploader onImageUpload={handleImageUpload} isLoading={isLoading} />
            </div>
          </div>
          <div className="w-full h-full flex flex-col gap-4">
            <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-cyan-400">AI Analysis</h2>
                { (imageUrl || error) && (
                    <button 
                      onClick={handleReset} 
                      className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm font-semibold transition-colors disabled:opacity-50"
                      disabled={isLoading}>
                      Start Over
                    </button>
                )}
            </div>
            <div className="bg-slate-800 rounded-lg p-4 flex-grow flex items-center justify-center min-h-[300px] lg:min-h-[400px] border-2 border-dashed border-slate-700">
              {isLoading && <Loader />}
              {error && !isLoading && (
                <div className="text-center text-red-400">
                  <IconAlertTriangle className="mx-auto h-12 w-12 mb-4" />
                  <p className="font-semibold">Analysis Failed</p>
                  <p className="text-sm max-w-sm mx-auto">{error}</p>
                </div>
              )}
              {!isLoading && !error && imageUrl && <AnalysisDisplay imageUrl={imageUrl} results={results} />}
              {!isLoading && !error && !imageUrl && (
                <div className="text-center text-slate-500">
                  <p className="text-lg font-medium">Results will be displayed here.</p>
                  <p className="text-sm">Upload a dental X-ray to begin the analysis.</p>
                </div>
              )}
            </div>

            { (imageUrl || (error && results)) && (
                <div className="bg-slate-800 rounded-lg p-4 border border-slate-700/50">
                    <h3 className="text-xl font-bold text-cyan-400 mb-3 flex items-center gap-2">
                        <IconSparkles className="w-6 h-6" />
                        <span>Report & Recommendations</span>
                    </h3>
                    {isLoading && (
                        <div className="flex items-center gap-3 text-slate-400 animate-pulse">
                            <div className="w-5 h-5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
                            <span>Generating report...</span>
                        </div>
                    )}
                    {!isLoading && analysisDescription && (
                        <AnalysisDescription description={analysisDescription} />
                    )}
                    {!isLoading && error && !analysisDescription && (
                        <p className="text-red-400 text-sm">Could not generate a report due to an earlier error.</p>
                    )}
                </div>
            )}
          </div>
        </div>
      </main>
      <footer className="text-center p-4 text-xs text-slate-500">
        <p>This tool is for informational purposes only and not a substitute for professional medical advice. Based on "Fusion of Image Filtering and Knowledge-Distilled YOLO Models for Root Canal Failure Diagnosis". Now enhanced with YOLO object detection for improved analysis.</p>
      </footer>
    </div>
  );
};

export default App;
