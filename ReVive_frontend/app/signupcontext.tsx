import React, { createContext, useContext, useState, ReactNode } from 'react';

// Define the shape of the context
interface SignupContextType {
  fromSignup: boolean;
  setFromSignup: (value: boolean) => void;
}

// Create context with a default value
const SignupContext = createContext<SignupContextType | undefined>(undefined);

// Hook to use the SignupContext
export const useSignup = (): SignupContextType => {
  const context = useContext(SignupContext);
  if (!context) {
    throw new Error('useSignup must be used within a SignupProvider');
  }
  return context;
};

// Provider component
export const SignupProvider = ({ children }: { children: ReactNode }) => {
  const [fromSignup, setFromSignup] = useState(false);

  return (
    <SignupContext.Provider value={{ fromSignup, setFromSignup }}>
      {children}
    </SignupContext.Provider>
  );
};
