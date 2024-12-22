import { Tabs } from 'expo-router';
import React from 'react';
import { Platform } from 'react-native';
import { View, Text, Image } from 'react-native';

import { useColorScheme } from '@/hooks/useColorScheme';

export default function TabLayout() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  return (
    <Tabs 
    screenOptions={{
      tabBarShowLabel: false,
      tabBarStyle: {
        position: 'absolute',
        elevation: 0,
        backgroundColor: isDark ? "#121212" : "#F5F5F5",
        height: 90,
      },
	  headerShown: false 
    }}>
      <Tabs.Screen
        name="home"
        options={{ 
          headerShown:false,
          tabBarIcon: ({focused}) => (
            <View style={{alignItems: 'center', justifyContent: 'center', top: 10}}>
              <Image
                source={isDark ? require('../../assets/images/darkhome.png') : require('../../assets/images/Home.png') }
                resizeMode='contain'
                style={{
                  tintColor: focused ? '#4169E1':'#e32f45',
                }}
              />
            </View>
          ), 
          }} />
        <Tabs.Screen name="profile" options={{ headerShown:false,
          tabBarIcon: ({focused}) => (
            <View style={{alignItems: 'center', justifyContent: 'center', top: 10}}>
              <Image
                source={isDark ? require('../../assets/images/darkprofile.png') : require('../../assets/images/Profile.png')}
                resizeMode='contain'
                style={{
                  tintColor: focused ? '#4169E1':'#e32f45',
                }}
              />
            </View>
          )  
        }}
      />
    </Tabs>
  );
}
