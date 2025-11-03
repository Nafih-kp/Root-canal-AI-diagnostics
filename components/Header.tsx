
import React from 'react';

export const Header: React.FC = () => {
  return (
    <header className="bg-slate-800/50 backdrop-blur-sm shadow-md sticky top-0 z-10">
      <div className="container mx-auto px-4 md:px-8 py-4 flex items-center">
         <svg className="w-8 h-8 mr-3 text-cyan-400" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17.5H11V16.5L13 14.5V11.5C13 10.12 11.88 9 10.5 9H10V7H14V10.5L12 12.5V13H13V17.5ZM10 14H12V15H10V14Z" fill="currentColor"/>
        </svg>
        <h1 className="text-xl md:text-2xl font-bold tracking-tight text-white">
          Root Canal AI Diagnostics
        </h1>
      </div>
    </header>
  );
};
