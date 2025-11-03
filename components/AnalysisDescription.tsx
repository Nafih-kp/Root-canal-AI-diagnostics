
import React from 'react';

interface AnalysisDescriptionProps {
  description: string;
}

export const AnalysisDescription: React.FC<AnalysisDescriptionProps> = ({ description }) => {
  return (
    <div className="text-sm text-gray-300 space-y-3 leading-relaxed">
      {description.split('\n').map((paragraph, index) => {
        const trimmed = paragraph.trim();
        if (trimmed.startsWith('### ')) {
            return <h3 key={index} className="text-md font-semibold text-cyan-300 mt-4 mb-1">{trimmed.substring(4)}</h3>
        }
        if (trimmed.startsWith('## ')) {
            return <h2 key={index} className="text-lg font-bold text-cyan-400 mt-5 mb-2">{trimmed.substring(3)}</h2>
        }
        if (trimmed.startsWith('- ')) {
             return <li key={index} className="ml-5 list-disc">{trimmed.substring(2)}</li>
        }
        if (trimmed.startsWith('**') && trimmed.endsWith('**')) {
            return <p key={index} className="font-bold my-2">{trimmed.substring(2, trimmed.length - 2)}</p>
        }
        if (trimmed === '') {
            return null; // Don't render empty paragraphs
        }
        return <p key={index}>{trimmed}</p>
      })}
    </div>
  );
};
