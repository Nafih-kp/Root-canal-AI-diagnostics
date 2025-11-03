
import React, { useState, useCallback, useRef } from 'react';
import { IconUpload } from './IconComponents';

interface ImageUploaderProps {
  onImageUpload: (base64: string, file: File) => void;
  isLoading: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageUpload, isLoading }) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File | null) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (typeof e.target?.result === 'string') {
          onImageUpload(e.target.result, file);
        }
      };
      reader.readAsDataURL(file);
    }
  }, [onImageUpload]);

  const onDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };
  
  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault(); // Necessary to allow drop
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };
  
  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0]) {
          handleFile(e.target.files[0]);
      }
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div
      className={`relative w-full h-full p-4 border-2 border-dashed rounded-lg flex flex-col items-center justify-center text-center transition-all duration-300 ${isDragging ? 'border-cyan-400 bg-slate-700/50' : 'border-slate-700 bg-slate-800'}`}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        onChange={onFileChange}
        accept="image/*"
        disabled={isLoading}
      />
      <IconUpload className={`w-16 h-16 mb-4 transition-colors ${isDragging ? 'text-cyan-400' : 'text-slate-500'}`} />
      <p className="mb-2 text-lg font-semibold text-gray-300">
        Drag & drop your X-ray image here
      </p>
      <p className="text-slate-400 mb-4">or</p>
      <button
        onClick={onButtonClick}
        disabled={isLoading}
        className="px-6 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Browse Files
      </button>
      <p className="text-xs text-slate-500 mt-4">Supports JPEG, PNG, WEBP</p>
    </div>
  );
};
