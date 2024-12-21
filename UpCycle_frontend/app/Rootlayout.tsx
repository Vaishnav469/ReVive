import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import { useState } from 'react';
import { useFonts } from 'expo-font';
import { Stack, router } from 'expo-router';
import { User, onAuthStateChanged } from 'firebase/auth';
import { FIREBASE_AUTH } from './firebaseconfig';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import { useEffect } from 'react';

import { useColorScheme } from '@/hooks/useColorScheme';


import { useSignup } from './signupcontext';


// Prevent the splash screen from auto-hiding before asset loading is complete.
SplashScreen.preventAutoHideAsync();

export default function RootLayout() {
  const [user, setUser] = useState<User | null>(null);
  const { fromSignup, setFromSignup } = useSignup();

  const colorScheme = useColorScheme();
  const [loaded] = useFonts({
    SpaceMono: require('@/assets/fonts/SpaceMono-Regular.ttf'),
  });

  useEffect(() => {
    if (loaded) {
      SplashScreen.hideAsync();
    }

    const unsubscribe = onAuthStateChanged(FIREBASE_AUTH, (user) => {
      console.log('user', user)
      console.log('fromSignup', fromSignup)
      if (user && fromSignup) {
        setUser(user);
        router.replace('/questions/Questionscreen?key=Location');
      } else if (user) {
        setUser(user);
        router.replace('/(tabs)/home');
      } else {
        router.replace('/SignUpLogin/SignUpLogin');
      }
    });
    return () => unsubscribe();

  }, [loaded, fromSignup]);

  if (!loaded) {
    return null;
  }

  return (
    <ThemeProvider value={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
      <Stack screenOptions={{headerShown: false}}>
          <Stack.Screen name="SignUpLogin/SignUpLogin" 
          options={{ headerShown:false }}
          />
          <Stack.Screen name="(tabs)" options={{ headerShown: false }} >
          </Stack.Screen>
          {/* <Stack.Screen name="+not-found" /> */}
      </Stack>
      <StatusBar style="auto" />
    </ThemeProvider>
  );
}
