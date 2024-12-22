// SignupLogin.js
import { useState } from 'react';
import { View, Text, TextInput, Button, ActivityIndicator, StyleSheet, KeyboardAvoidingView } from 'react-native';
import { FIRESTORE_DB, FIREBASE_AUTH } from '../firebaseconfig';
import { doc, setDoc } from 'firebase/firestore';
import { signInWithEmailAndPassword, createUserWithEmailAndPassword } from 'firebase/auth';
import { useSignup } from '../signupcontext';
import { router } from 'expo-router';
import { useColorScheme } from '@/hooks/useColorScheme';

const SignupLogin = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [isSigningUp, setIsSigningUp] = useState(false); // Toggle between sign up and login
  const [loading, setloading] = useState(false)
  const { setFromSignup } = useSignup(); // Access the state updater from context
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  const auth = FIREBASE_AUTH;

  const handleSignup = async () => {
    setloading(true);
    try {
      setFromSignup(true);
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;
      await setDoc(doc(FIRESTORE_DB, 'profile', user.uid), { username:username, email:email });

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
      setFromSignup(false);
      await signInWithEmailAndPassword(auth, email, password);
    } catch (error) {
      console.error('Login Error:', error);
    } finally {
      setloading(false);
    }
  };

  return (
    <View style={[styles.container, {backgroundColor: isDark ? "#121212" : "#F3ECF4"}]}>
      <KeyboardAvoidingView behavior='padding'>
        <Text style={[styles.title, { color: isDark ? "#FFFFFF" : "#000000"  }]}>{isSigningUp ? 'Sign Up' : 'Log In'}</Text>
            {isSigningUp && (<TextInput
            style={[styles.input, {
              color: isDark ? "#FFFFFF" : "#000000",
              borderColor: isDark ? "#3E3E3E" : "#CCCCCC",
              backgroundColor: isDark ? "#1E1E1E" : "#F5F5F5",
          }]}
            placeholder="Username"
            placeholderTextColor={isDark ? "#AAAAAA" : "#888888"}
            value={username}
            onChangeText={(e) => setUsername(e)}
            autoCapitalize='none'
            />)
            }
           
            <TextInput
                style={[styles.input, {
                  color: isDark ? "#FFFFFF" : "#000000",
                  borderColor: isDark ? "#3E3E3E" : "#CCCCCC",
                  backgroundColor: isDark ? "#1E1E1E" : "#F5F5F5",
              }]}
                placeholder="Email"
                placeholderTextColor={isDark ? "#AAAAAA" : "#888888"}
                value={email}
                onChangeText={(e) => setEmail(e)}
                autoCapitalize='none'
            />
            <TextInput
                style={[styles.input, {
                  color: isDark ? "#FFFFFF" : "#000000",
                  borderColor: isDark ? "#3E3E3E" : "#CCCCCC",
                  backgroundColor: isDark ? "#1E1E1E" : "#F5F5F5",
              }]}
                placeholder="Password"
                placeholderTextColor={isDark ? "#AAAAAA" : "#888888"}
                secureTextEntry
                value={password}
                onChangeText={(Text) => setPassword(Text)}
                autoCapitalize='none'
            />

            {loading ? (<ActivityIndicator size="large" color="#0000ff" />) :
                (<Button
                    color={isDark ? "#BB86FC" : "#6200EE"}
                    title={isSigningUp ? "Sign Up" : "Log In"}
                    onPress={isSigningUp ? handleSignup : handleLogin}
                />)
            }
            <Button
                color={isDark ? "#BB86FC" : "#6200EE"}
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
    }
  });

export default SignupLogin;
