
import React, { useState, useRef, useEffect } from 'react';
import type { DetectionResult } from '../types';
import { CLASS_COLORS } from '../constants';

interface AnalysisDisplayProps {
  imageUrl: string;
  results: DetectionResult[] | null;
}

export const AnalysisDisplay: React.FC<AnalysisDisplayProps> = ({ imageUrl, results }) => {
  const imageRef = useRef<HTMLImageElement>(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const updateSize = () => {
      if (imageRef.current) {
        setImageSize({
          width: imageRef.current.offsetWidth,
          height: imageRef.current.offsetHeight,
        });
      }
    };

    const imgElement = imageRef.current;
    if (imgElement) {
      imgElement.addEventListener('load', updateSize);
      window.addEventListener('resize', updateSize);
      
      // If image is already loaded (e.g., from cache)
      if (imgElement.complete) {
        updateSize();
      }
    }

    return () => {
      if (imgElement) {
        imgElement.removeEventListener('load', updateSize);
      }
      window.removeEventListener('resize', updateSize);
    };
  }, [imageUrl]);

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      <img
        ref={imageRef}
        src={imageUrl}
        alt="Dental radiograph"
        className="max-w-full max-h-full object-contain rounded-md"
      />
      {results && imageSize.width > 0 && (
        <div className="absolute top-0 left-0" style={{ width: imageSize.width, height: imageSize.height }}>
          {results.map((result, index) => {
            const { x, y, width, height } = result.box;
            const colorConfig = CLASS_COLORS[result.label] || CLASS_COLORS['default'];
            const confidencePercent = (result.confidence * 100).toFixed(1);
            
            return (
              <div
                key={index}
                className={`absolute ${colorConfig.border} border-2 rounded-md shadow-lg`}
                style={{
                  left: `${x * 100}%`,
                  top: `${y * 100}%`,
                  width: `${width * 100}%`,
                  height: `${height * 100}%`,
                }}
              >
                <div className={`absolute -top-6 left-0 text-xs font-semibold px-2 py-0.5 rounded-t-md text-white ${colorConfig.bg}`}>
                  {result.label} ({confidencePercent}%)
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
