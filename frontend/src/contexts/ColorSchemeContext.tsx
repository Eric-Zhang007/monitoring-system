import { createContext, useContext, useState, ReactNode } from 'react';
import { ColorScheme, ColorSchemeContextType } from '../types';

const ColorSchemeContext = createContext<ColorSchemeContextType | undefined>(undefined);

export function ColorSchemeProvider({ children }: { children: ReactNode }) {
  const [scheme, setScheme] = useState<ColorScheme>('cn');

  const toggleScheme = () => {
    setScheme(prev => prev === 'cn' ? 'us' : 'cn');
  };

  return (
    <ColorSchemeContext.Provider value={{ scheme, toggleScheme }}>
      {children}
    </ColorSchemeContext.Provider>
  );
}

export function useColorScheme() {
  const context = useContext(ColorSchemeContext);
  if (context === undefined) {
    throw new Error('useColorScheme must be used within a ColorSchemeProvider');
  }
  return context;
}
