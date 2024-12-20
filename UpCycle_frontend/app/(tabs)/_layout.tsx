import { Tabs } from 'expo-router';
import React from 'react';
import { Platform } from 'react-native';
import { View, Text, Image } from 'react-native';

import { useColorScheme } from '@/hooks/useColorScheme';
import FontAwesome from '@expo/vector-icons/FontAwesome';

export default function TabLayout() {
  const colorScheme = useColorScheme();
  return (
    <Tabs 
    screenOptions={{
      tabBarShowLabel: false,
      tabBarStyle: {
        position: 'absolute',
        elevation: 0,
        backgroundColor: '#fff',
        height: 90,
      },
	  headerShown: false 
    }}>
      {/* <Tabs.Screen
        name="home"
        options={{ 
          headerShown:false,
          tabBarIcon: ({focused}) => (
            <View style={{alignItems: 'center', justifyContent: 'center', top: 10}}>
              <Image
                source={require('../../assets/images/Home.png')}
                resizeMode='contain'
                style={{
                  tintColor: focused ? '#4169E1':'#e32f45',
                }}
              />
              <Text style={{color: focused ? '#4169E1':'#e32f45', fontSize: 12}}>Home</Text>
            </View>
          ), 
          }} />
        <Tabs.Screen name="explore" options={{ 
          headerShown:false,
          tabBarIcon: ({focused}) => (
            <View style={{alignItems: 'center', justifyContent: 'center', top: 10}}>
              <Image
                style={{flex: 1,
                  tintColor: focused ? '#4169E1':'#e32f45',
                }}
                source={require('../../assets/images/search.png')}
                resizeMode='contain'
              />
              <Text style={{color: focused ? '#4169E1':'#e32f45', fontSize: 12}}>Search</Text>
            </View>
          ), 
          }} />
        <Tabs.Screen name="leaderboard" options={{ 
          headerShown:false,
          tabBarIcon: ({focused}) => (
            <View style={{alignItems: 'center', justifyContent: 'center', top: 10}}>
              <Image
                source={require('../../assets/images/Leaderboard.png')}
                resizeMode='contain'
                style={{
                  tintColor: focused ? '#4169E1':'#e32f45',
                }}
              />
              <Text style={{color: focused ? '#4169E1':'#e32f45', fontSize: 12}}>Leaderboard</Text>
            </View>
          ), 
          }} /> */}
        <Tabs.Screen name="profile" options={{ headerShown:false,
          tabBarIcon: ({focused}) => (
            <View style={{alignItems: 'center', justifyContent: 'center', top: 10}}>
              <Image
                source={require('../../assets/images/Profile.png')}
                resizeMode='contain'
                style={{
                  tintColor: focused ? '#4169E1':'#e32f45',
                }}
              />
              <Text style={{color: focused ? '#4169E1':'#e32f45', fontSize: 12}}>Profile</Text>
            </View>
          )  
        }}
      />
    </Tabs>
  );
}
