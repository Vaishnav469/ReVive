import React from 'react';
import { SignupProvider } from './signupcontext'; // Make sure the path is correct
import RootLayout from './Rootlayout'; // Path to RootLayout
import { ThemeProvider } from '@react-navigation/native';
import { DarkTheme, DefaultTheme } from '@react-navigation/native';
import { useColorScheme } from '@/hooks/useColorScheme';

export default function Layout() {
    const colorScheme = useColorScheme();
  return (
    <SignupProvider>
      <ThemeProvider value={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
        <RootLayout />
      </ThemeProvider>
    </SignupProvider>
  );
}
