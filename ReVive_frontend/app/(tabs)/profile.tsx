import React, { useState, useEffect } from 'react';
import { View, StyleSheet, Text, ActivityIndicator } from 'react-native';
import { Button, Card, Avatar } from '@rneui/themed';
import { doc, getDoc } from 'firebase/firestore';
import { FIRESTORE_DB, FIREBASE_AUTH } from '../firebaseconfig';
import { useColorScheme } from '@/hooks/useColorScheme';
import * as Font from 'expo-font';

const fetchFonts = () => {
  return Font.loadAsync({
    'LemonJelly': require('../../assets/fonts/LemonJelly.ttf'),
  });
};

const Profile = () => {
  const [user, setUser] = useState(FIREBASE_AUTH.currentUser);
  const [profileData, setProfileData] = useState<any>(null); 
  const [loading, setloading] = useState(false)
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  
  const [fontLoaded, setFontLoaded] = useState(false);
  useEffect(() => {
    (async () => {
      try {
        await fetchFonts();
        setFontLoaded(true);
      } catch (error) {
        console.error("Error loading fonts: ", error);
      }
    })();
  }, []);

  useEffect(() => {
    const fetchProfileData = async () => {
      setloading(true);
      try {
        if(user) {
          const document = await getDoc(doc(FIRESTORE_DB, 'profile', user.uid));
          if (document.exists()) {
              setProfileData(document.data());
            }
        }
        
      } catch (err) {
        console.error("Error fetching profile data: ", err);
      } finally {
        setloading(false);
      }
    };
    fetchProfileData();
  }, []);

  return (
    loading ?  <View style={[styles.container]}><ActivityIndicator size="large" color="#0000ff"  /></View> :
    user && profileData && (
      <View style={[styles.scrollContainer, {backgroundColor: isDark ? "#121212" : "#F3ECF4"}]}>
      <Text style={[styles.title, { color: isDark ? "#FFFFFF" : "#000000"  }]}>ReVive</Text>
      <View style={[styles.container]}>
      <Card containerStyle={[styles.card, {backgroundColor: isDark ? "#121212" : "#F3ECF4"}]}>
        <View style={[styles.header]}>
          <Avatar
            size="large"
            rounded
            source={{ uri: user.photoURL || 'https://placeimg.com/140/140/any' }}
          />
          <Text style={[styles.username,  { color: isDark ? "#FFFFFF" : "#000000"  }]}>{profileData.username || 'User Name'}</Text>
        </View>
        <View style={styles.details}>
          <Text style={styles.label}>Location:</Text>
          <Text style={[styles.info,  { color: isDark ? "#FFFFFF" : "#000000"  }]}>{profileData.Location}</Text>
          <Text style={styles.label}>Prefer ideas that require money:</Text>
          <Text style={[styles.info,  { color: isDark ? "#FFFFFF" : "#000000"  }]}>{profileData.Money}</Text>
          <Text style={styles.label}>Time for Upcycling:</Text>
          <Text style={[styles.info,  { color: isDark ? "#FFFFFF" : "#000000"  }]}>{profileData.Time}</Text>
        </View>
        <Button
          buttonStyle={styles.button}
          title="Logout"
          onPress={() => FIREBASE_AUTH.signOut()}
        />
      </Card>
    </View>
    </View>
    )
  );
};

const styles = StyleSheet.create({
  scrollContainer: {
    flexGrow: 1,
    backgroundColor: '#fff',
  },
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    fontSize: 43,
    fontWeight: 'bold',
    fontFamily: 'LemonJelly',
    top: 63,
    left: 30,
    marginBottom: 20,
  },
  card: {
    width: '90%',
    borderRadius: 10,
    padding: 20,
    backgroundColor: '#ffffff',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  username: {
    fontSize: 24,
    fontWeight: 'bold',
    marginLeft: 20,
  },
  details: {
    marginBottom: 20,
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#555',
  },
  info: {
    fontSize: 16,
    marginBottom: 10,
  },
  button: {
    backgroundColor: '#4169E1',
  },
});

export default Profile;
