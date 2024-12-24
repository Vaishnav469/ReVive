import { Stack, router } from "expo-router";
import React, { useState, createContext, useContext, useEffect } from 'react';
import { FIRESTORE_DB, FIREBASE_AUTH } from '../firebaseconfig';
import { doc, updateDoc } from 'firebase/firestore';
import QuestionScreen from './Questionscreen';
import { useSignup } from '../signupcontext';

const questions = [
  { key: 'Location', question: "Where are you located at? (for personalized suggestions" },
  { key: 'Money', question: "Prefer ideas that require money? (Yes/No)" },
  { key: 'Time', question: "How much time do you like to spend Upcycling? (Ex: 1 hour/month)" },
];

// Define the context type
interface OnboardingContextType {
  handleResponse: (key: string, response: string) => void;
  questions: { key: string; question: string }[];
}

// Create the context with undefined as the default value
const OnboardingContext = createContext<OnboardingContextType | undefined>(undefined);

export const useOnboarding = () => {
  const context = useContext(OnboardingContext);
  if (!context) {
    throw new Error("useOnboarding must be used within an OnboardingProvider");
  }
  return context;
};


export default function QuestionLayout() {
  const auth = FIREBASE_AUTH;
  const [responses, setResponses] = useState({});
  const { setFromSignup } = useSignup(); 

  const handleResponse = async (key: string, response: string) => {
    setResponses((prev) => {
      const updatedResponses = { ...prev, [key]: response };
      const nextIndex = questions.findIndex(q => q.key === key) + 1;
      if (nextIndex < questions.length) {
        // Using router.push for navigation in Expo Router
        router.push(`/questions/Questionscreen?key=${questions[nextIndex].key}`);
      } else {
        const user = auth.currentUser;
        if (user) {
          updateDoc(doc(FIRESTORE_DB, 'profile', user.uid),  updatedResponses)
            .then(() => {
              setFromSignup(false);
            })
            .catch((error) => {
              console.error("Error saving responses:", error);
            })
        }
      }
      return updatedResponses;
    });
  };


  return (
    <OnboardingContext.Provider value={{ handleResponse, questions }}>
      <Stack screenOptions={{ headerShown: false }}>
        <Stack.Screen name="Questionscreen" options={{ headerShown: false }} />
      </Stack>
    </OnboardingContext.Provider>
  )
}
