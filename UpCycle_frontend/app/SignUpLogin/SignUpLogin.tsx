// SignupLogin.js
import { useState } from 'react';
import { View, Text, TextInput, Button, ActivityIndicator, StyleSheet, KeyboardAvoidingView } from 'react-native';
import { FIREBASE_AUTH } from '../firebaseconfig';

import { signInWithEmailAndPassword, createUserWithEmailAndPassword } from 'firebase/auth';
import { useSignup } from '../signupcontext';
import { router } from 'expo-router';

const SignupLogin = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [isSigningUp, setIsSigningUp] = useState(false); // Toggle between sign up and login
  const [loading, setloading] = useState(false)
  const { fromSignup, setFromSignup } = useSignup(); // Access the state updater from context

  
  
  const auth = FIREBASE_AUTH;

  const handleSignup = async () => {
    setloading(true);
    try { 
      console.log(email, password)
      setFromSignup(true);
      await createUserWithEmailAndPassword(auth, email, password);
      await signInWithEmailAndPassword(auth, email, password);
      

    } catch (error) {
      console.error('Signup Error:', error);
    } finally {
      setloading(false);
    }
  };

  const handleLogin = async () => {
    setloading(true);
    try {
      console.log(email, password)
      setFromSignup(false);
      await signInWithEmailAndPassword(auth, email, password);
    } catch (error) {
      console.error('Login Error:', error);
    } finally {
      setloading(false);
    }
  };

  return (
    <View style={styles.container}>
      <KeyboardAvoidingView behavior='padding'>
        <Text style={styles.title}>{isSigningUp ? 'Sign Up' : 'Log In'}</Text>
            {isSigningUp && (<TextInput
            style={styles.input}
            placeholder="Username"
            placeholderTextColor="#666" 
            value={username}
            onChangeText={(e) => setUsername(e)}
            autoCapitalize='none'
            />)
            }
           
            <TextInput
                style={styles.input}
                placeholder="Email"
                placeholderTextColor="#666" 
                value={email}
                onChangeText={(e) => setEmail(e)}
                autoCapitalize='none'
            />
            <TextInput
                style={styles.input}
                placeholder="Password"
                placeholderTextColor="#666" 
                secureTextEntry
                value={password}
                onChangeText={(Text) => setPassword(Text)}
                autoCapitalize='none'
            />

            {loading ? (<ActivityIndicator size="large" color="#0000ff" />) :
                (<Button
                    title={isSigningUp ? "Sign Up" : "Log In"}
                    onPress={isSigningUp ? handleSignup : handleLogin}
                />)
            }
            <Button
                title={`Switch to ${isSigningUp ? "Log In" : "Sign Up"}`}
                onPress={() => setIsSigningUp(!isSigningUp)}
            />
        </KeyboardAvoidingView>
    </View>
  );
};
const styles = StyleSheet.create({
    container: {
      flex: 1,
      justifyContent: 'center',
      padding: 20,
      backgroundColor: '#F3ECF4',
    },
    title: {
      fontSize: 24,
      fontWeight: 'bold',
      marginBottom: 20,
      textAlign: 'center',
    },
    input: {
      height: 40,
      borderWidth: 1,
      borderColor: '#ddd',
      borderRadius: 5,
      marginBottom: 15,
      paddingHorizontal: 10,
      backgroundColor: '#fff',
    }
  });

export default SignupLogin;
