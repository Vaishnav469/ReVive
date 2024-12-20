// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
// @ts-ignore 
import { initializeAuth } from "firebase/auth";
import { getReactNativePersistence } from '@firebase/auth/dist/rn/index.js';
import ReactNativeAsyncStorage from '@react-native-async-storage/async-storage';
import { getFirestore } from "firebase/firestore";


// Your web app's Firebase configuration
const firebaseConfig = {
    apiKey: 'AIzaSyC3StBlEeyVyLKSIJiXMp7OoZwa4rdcUI8',
    authDomain: "upcycle-f1725.firebaseapp.com",
    projectId: "upcycle-f1725",
    storageBucket: "upcycle-f1725.firebasestorage.app",
    messagingSenderId: "346546567314",
    appId: "1:346546567314:web:6c4ca76fe04d1398f4ce4b"
};

// Initialize Firebase
export const FIREBASE_APP = initializeApp(firebaseConfig);

export const FIREBASE_AUTH = initializeAuth(FIREBASE_APP, {
  persistence: getReactNativePersistence(ReactNativeAsyncStorage)
});

export const FIRESTORE_DB = getFirestore(FIREBASE_APP); 
