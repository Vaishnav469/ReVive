import React, { useState, useEffect } from 'react';
import { View, Text, Button, Image, StyleSheet,KeyboardAvoidingView, Platform, TextInput,ScrollView, ActivityIndicator, Alert, FlatList, TouchableOpacity, Linking   } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import { useTheme } from "@react-navigation/native";
import { useColorScheme } from '@/hooks/useColorScheme';
import { doc, getDoc } from 'firebase/firestore';
import { FIRESTORE_DB, FIREBASE_AUTH } from '../firebaseconfig';

export default function HomePage() {
  const [image, setImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [ideas, setIdeas] = useState(null);
  const [manualInput, setManualInput] = useState('');
  const [cameraPermission, setCameraPermission] = useState(false);
  const { colors } = useTheme();
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  const [loading, setloading] = useState(false)
  const [profileData, setProfileData] = useState<any>(null); 
  const auth = FIREBASE_AUTH;
  

  // Request permissions for camera and media library
  useEffect(() => {
    (async () => {
      const cameraStatus = await ImagePicker.requestCameraPermissionsAsync();
      const mediaLibraryStatus = await ImagePicker.requestMediaLibraryPermissionsAsync();
      setCameraPermission(cameraStatus.granted && mediaLibraryStatus.granted);

      if (!cameraStatus.granted || !mediaLibraryStatus.granted) {
        Alert.alert('Permissions required', 'Camera and Media Library permissions are required to use this feature.');
      }
      const fetchProfileData = async () => {
        try {
          const user = auth.currentUser;
          if(user) {
            const document = await getDoc(doc(FIRESTORE_DB, 'profile', user.uid));
            if (document.exists()) {
                setProfileData(document.data());
              } else {
                console.log("No such document!");
            }
          }
          
        } catch (err) {
          console.error("Error fetching profile data: ", err);
        }
      };
      fetchProfileData();
    })();
  }, []);

  const pickImage = async () => {
    if (!cameraPermission) {
      Alert.alert('Permission Denied', 'Please grant media library permissions to select an image.');
      return;
    }

    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  const takePhoto = async () => {
    if (!cameraPermission) {
      Alert.alert('Permission Denied', 'Please grant camera permissions to take a photo.');
      return;
    }

    let result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  const submitImage = async () => {
    if (!image) return;

    const formData = new FormData();
    formData.append('file', { uri: image, name: 'photo.jpg', type: 'image/jpeg' });
    formData.append('location', profileData.Location);
    formData.append('money', profileData.Money);
    formData.append('time', profileData.Time);
    setloading(true);
    

    try {
      const response = await axios.post('http://192.168.70.47:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setPrediction(response.data.prediction);
      setIdeas(response.data.upcycling_ideas);
    } catch (err) {
      console.error(err);
    } finally {
        setloading(false);
    }
  };

    const handleManualSubmit = async () => {
        if (!manualInput.trim()) {
            Alert.alert('Error', 'Please enter an item name.');
            return;
    }

    setloading(true);
    try {
      const response = await axios.post('http://192.168.70.47:8000/manual-input', { 
        item_name: manualInput,
        ocation: profileData.Location,
        money: profileData.Money,
        time: profileData.Time 
    });
      setPrediction(manualInput);
      setIdeas(response.data.upcycling_ideas);
      setManualInput('');
    } catch (err) {
      console.error(err);
      Alert.alert('Error', 'Failed to fetch upcycling ideas. Please try again.');
    } finally {
        setloading(false);
    }
  };

  const renderIdea = ({ item }: { item: any }) => (
    <View style={[styles.ideaCard, {  backgroundColor: isDark ? "#1E1E1E" : "#FFFFFF", borderColor: isDark ? "#3E3E3E" : "#CCCCCC", }]}>
      <Text style={[styles.ideaTitle, { color: isDark ? "#FFFFFF" : "#000000" }]}>{item.title}</Text>
      <Text style={[styles.ideaDescription, { color: isDark ? "#CCCCCC" : "#444444", marginBottom: 40 }]}>{item.description}</Text>
      {item.videos && item.videos.map((video: any, videoIndex: number) => (
                  <View key={videoIndex} style={[styles.videoContainer, {  backgroundColor: isDark ? "#1E1E1E" : "#F5F5F5" }]}>
                    <TouchableOpacity onPress={() => Linking.openURL(video.url)}>
                      <Image source={{ uri: video.thumbnail }} style={styles.thumbnail} />
                    </TouchableOpacity>
                    <Text style={[styles.videoTitle, { color: isDark ? "#FFFFFF" : "#000000" }]}>{video.title}</Text>
                  </View>
                ))}
    </View>
  );

  const resetImage = () => {
    setImage(null);
    setPrediction(null);
    setIdeas(null);
    setManualInput('');
  };

  return (
    <KeyboardAvoidingView
    style={{ flex: 1 }}
    behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    keyboardVerticalOffset={Platform.OS === 'ios' ? 50 : 0} // Adjust offset as needed
  >
    <ScrollView contentContainerStyle={[styles.scrollContainer,  { backgroundColor: colors.background }]}>
        <View style={styles.container}>
            <Text style={[styles.title, { color: isDark ? "#FFFFFF" : "#000000"  }]}>Upcycle</Text>
            {prediction === null && (
            <View style={styles.buttonContainer}>
                <Button  color={isDark ? "#BB86FC" : "#6200EE"} title="Pick an Image" onPress={pickImage} />
                <Button  color={isDark ? "#BB86FC" : "#6200EE"} title="Take a Photo" onPress={takePhoto} />
            </View>)}
            {image && <Image source={{ uri: image }} style={styles.image} />}
            {loading && prediction === null ? <View style={styles.button}>
                <ActivityIndicator size="large" color="#0000ff"  />
                </View> : image && prediction === null && (
                <View style={styles.button}>
                    <Button color={isDark ? "#BB86FC" : "#6200EE"} title="Submit Image" onPress={submitImage} />
                </View>
            )}
            

            {prediction && ideas && (
                <View style={[styles.container]}>
                    <Text style={[styles.prediction, { color: isDark ? "#FFFFFF" : "#000000" }]}>Prediction: {prediction}</Text>
                    <Text style={[styles.ideas, { color: isDark ? "#CCCCCC" : "#444444" }]}>Upcycling Ideas:</Text>
                    <FlatList
                        data={ideas}
                        keyExtractor={(item, index) => index.toString()}
                        renderItem={renderIdea}
                        horizontal
                        showsHorizontalScrollIndicator={false}
                    />
                    <TextInput
                        style={[styles.input,
                            {
                                color: isDark ? "#FFFFFF" : "#000000",
                                borderColor: isDark ? "#3E3E3E" : "#CCCCCC",
                                backgroundColor: isDark ? "#1E1E1E" : "#F5F5F5",
                            },
                        ]}
                        placeholder="Not a correct prediction? Type the item name Here"
                        value={manualInput}
                        onChangeText={setManualInput}
                        placeholderTextColor={isDark ? "#AAAAAA" : "#888888"}
                    />
                    {loading ? 
                    <View style={styles.button2}>
                        <ActivityIndicator size="large" color="#0000ff"  />
                    </View> : 
                    manualInput &&  
                    <View style={styles.button2}>
                        <Button title="Submit" onPress={handleManualSubmit} />
                    </View>}

                    <View style={styles.button1}>
                        <Button  color={isDark ? "#BB86FC" : "#6200EE"} title="Want to see ideas for another item?" onPress={resetImage} />
                    </View>
          
                </View>
            )}
      </View>
      
    </ScrollView>
    </KeyboardAvoidingView>
    
  );
}

const styles = StyleSheet.create({
    scrollContainer: {
        flexGrow: 1,
        backgroundColor: '#fff',
      },
      container: {
        flex: 1,
        alignItems: 'center',
        padding: 20,
        width: 400
      },
      button: {
        marginTop: 100,
      },
      button1: {
        marginBottom: 60,
      },
      button2: {
        marginBottom: 20,
      },
      ideaCard: {
        width: 300,
        padding: 30,
        borderRadius: 10,
        marginHorizontal: 10,
        alignItems: 'center',
        shadowColor: '#000',
        shadowOpacity: 0.1,
        shadowOffset: { width: 0, height: 2 },
        shadowRadius: 5,
      },
      ideaDescription: {
        fontSize: 12,
        textAlign: 'center',
      },
      title: {
        fontSize: 24,
        fontWeight: 'bold',
        top: 40,
        marginBottom: 20,
      },
      buttonContainer: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 20,
        top: 300,
      },
      image: {
        width: 200,
        height: 200,
        marginBottom: 20,
        marginTop: 30,
      },
      resultContainer: {
        width: '100%',
      },
      prediction: {
        fontSize: 18,
        fontWeight: 'bold',
        marginBottom: 10,
      },
      ideas: {
        fontSize: 16,
        marginBottom: 10,
      },
      ideaContainer: {
        marginBottom: 20,
      },
      ideaTitle: {
        fontSize: 16,
        fontWeight: 'bold',
        marginBottom: 10,
      },
      videoContainer: {
        marginBottom: 40,
        alignItems: 'center',
      },
      thumbnail: {
        width: 150,
        height: 100,
        marginBottom: 5,
      },
      videoTitle: {
        fontSize: 14,
        textAlign: 'center',
      },
      input: {
        borderColor: '#ccc',
        borderWidth: 1,
        borderRadius: 5,
        padding: 20,
        marginTop: 30,
        marginBottom: 30,
      },
});
